""" Common math functions. """

import numpy as np

from RMS.Routines.SphericalPolygonCheck import sphericalPolygonCheck

def lineFunc(x, m, k):
    """ Linear function.
    
    Arguments:
        x: [float or ndarray] Independent variable.
        m: [float] Slope.
        k: [float] Y-intercept.

    Return:
        y: [float or ndarray] Dependant variable.
    """

    return m*x + k


def logLineFunc(x, m, k):
    """ Logarithmic linear function.

    Arguments:
        x: [float or ndarray] Independent variable.
        m: [float] Slope.
        k: [float] Y-intercept.

    Return:
        y: [float or ndarray] Dependant variable.
    """

    return 10**lineFunc(x, m, k)


def angularSeparation(ra1, dec1, ra2, dec2):
    """ Calculates the angle between two points on a sphere. Inputs in radians.
    
    Arguments:
        ra1: [float] Right ascension 1 (radians).
        dec1: [float] Declination 1 (radians).
        ra2: [float] Right ascension 2 (radians).
        dec2: [float] Declination 2 (radians).

    Return:
        [float] Angle between two coordinates (radians).
    """

    # Classical method
    return np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2 - ra1))

    # # Compute the angular separation using the haversine formula
    # #   Source: https://idlastro.gsfc.nasa.gov/ftp/pro/astro/gcirc.pro
    # deldec2 = (dec2 - dec1)/2.0
    # delra2 =  (ra2 - ra1)/2.0
    # sindis = np.sqrt(np.sin(deldec2)*np.sin(deldec2) \
    #     + np.cos(dec1)*np.cos(dec2)*np.sin(delra2)*np.sin(delra2))
    # dis = 2.0*np.arcsin(sindis) 

    # return dis

def angularSeparationDeg(ra1, dec1, ra2, dec2):
    """ Calculates the angle between two points on a sphere in degrees.

    Arguments:
        ra1: [float] Right ascension 1 (degrees).
        dec1: [float] Declination 1 (degrees).
        ra2: [float] Right ascension 2 (degress).
        dec2: [float] Declination 2 (degrees).

    Return:
        [float] Angle between two coordinates (degrees).
    """

    ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
    ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)


    return np.degrees(angularSeparation(ra1_rad, dec1_rad, ra2_rad, dec2_rad))


def angularSeparationVect(vect1, vect2):
    """ Calculates angle between vectors in radians. """

    return np.abs(np.arccos(np.dot(vect1, vect2)))



### VECTORS ###
##############################################################################################################


def vectNorm(vect):
    """ Convert a given vector to a unit vector. """

    return vect/vectMag(vect)



def vectMag(vect):
    """ Calculate the magnitude of the given vector. """

    return np.linalg.norm(vect)


##############################################################################################################


def rotatePoint(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.

    Source: http://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python

    Arguments:
        origin: [tuple of floats] (x, y) pair of Cartesian coordinates of the origin
        point: [tuple of floats] (x, y) pair of Cartesian coordinates of the point
        angle: [float] angle of rotation in radians

    Return:
        (qx, qy): [tuple of floats] Cartesian coordinates of the rotated point
    """

    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle)*(px - ox) - np.sin(angle)*(py - oy)
    qy = oy + np.sin(angle)*(px - ox) + np.cos(angle)*(py - oy)

    return qx, qy


def cartesianToPolar(x, y, z):
    """ Converts 3D cartesian coordinates to polar coordinates. 

    Arguments:
        x: [float] Px coordinate.
        y: [float] Py coordinate.
        z: [float] Pz coordinate.

    Return:
        (theta, phi): [float] Polar angles in radians (inclination, azimuth).

    """

    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    return theta, phi



def polarToCartesian(theta, phi):
    """ Converts 3D spherical coordinates to 3D cartesian coordinates. 

    Arguments:
        theta: [float] Longitude in radians.
        phi: [float] Latitude in radians.

    Return:
        (x, y, z): [tuple of floats] Coordinates of the point in 3D cartesian coordinates.
    """


    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    return x, y, z


def isAngleBetween(left, ang, right):
    """ Checks if ang is between the angle on the left and right. 
    
    Arguments:
        left: [float] Left (counter-clockwise) angle (radians).
        ang: [float] Angle to check (radians),
        right: [float] Right (clockwise) angle (radiant).

    Return:
        [bool] True if the angle is in between, false otherwise.
    """

    if right - left < 0:
        right = right - left + 2*np.pi
    else:
        right = right - left


    if ang - left < 0:
        ang = ang - left + 2*np.pi
    else:
        ang = ang - left


    return ang < right



@np.vectorize
def sphericalPointFromHeadingAndDistance(ra1, dec1, heading, distance):
    """ Given RA and Dec, a heading and angular distance, compute coordinates of the point.

    Arguments:
        ra1: [float] Right Ascension (deg).
        dec1: [float] Declination (deg).
        heading: [float] Heading +E of due N in degrees (deg).
        distance: [float] Distance (deg).

    Return:
        ra, dec: [float] Coordinates of the new point (deg)

    """

    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)
    heading = np.radians(heading)
    distance = np.radians(distance)

    # Compute the new declination
    dec = np.arcsin(np.sin(dec1)*np.cos(distance) + np.cos(dec1)*np.sin(distance)*np.cos(heading))

    # Compute the new RA
    dra = np.arctan2(np.sin(heading)*np.sin(distance)*np.cos(dec1), np.cos(distance) \
        - np.sin(dec1)*np.sin(dec))
    ra = (ra1 - dra + np.pi)%(2*np.pi) - np.pi


    return np.degrees(ra)%360, np.degrees(dec)
    


def RMSD(x, weights=None):
    """ Root-mean-square deviation of measurements vs. model. 
    
    Arguments:
        x: [ndarray] An array of model and measurement differences.

    Return:
        [float] RMSD
    """

    if isinstance(x, list):
        x = np.array(x)

    if weights is None:
        weights = np.ones_like(x)

    return np.sqrt(np.sum(weights*x**2)/np.sum(weights))


### 3D functions ###
##############################################################################################################

def sphericalToCartesian(r, theta, phi):
    """ Convert spherical coordinates to cartesian coordinates. 
        
    Arguments:
        r: [float] Radius
        theta: [float] Inclination in radians.
        phi: [float] Azimuth angle in radians.

    Return:
        (x, y, z): [tuple of floats] Coordinates of the point in 3D cartesian coordinates.
    """

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return x, y, z


def pointInsideConvexPolygonSphere(points, vertices):
    """
    LEGACY FUNCTION.

    Polygon must be convex
    https://math.stackexchange.com/questions/4012834/checking-that-a-point-is-in-a-spherical-polygon


    Arguments:
        points: [array] Points with dimension (npoints, 2). The two dimensions are ra and dec in degrees.
        vertices: [array] Vertices of convex polygon with dimension (nvertices, 2). The two dimensions are 
            ra and dec in degrees.
        
    Return:
        filter: [array of bool] Array of booleans on whether a given point is inside the polygon on 
            the sphere.
    """

    # Call the new function to check if the points are inside the polygon
    return sphericalPolygonCheck(vertices, points)


##############################################################################################################

def histogramEdgesEqualDataNumber(x, nbins):
    """ Given the data, divide the histogram edges in such a way that every bin has the same number of
        data points. 

        Source: https://stackoverflow.com/questions/37649342/matplotlib-how-to-make-a-histogram-with-bins-of-equal-area/37667480

    Arguments:
        x: [list] Input data.
        nbins: [int] Number of bins.

    """

    npt = len(x)
    return np.interp(np.linspace(0, npt, nbins + 1), np.arange(npt), np.sort(x))


def histogramEdgesDataNumber(x, points_per_bin):
    """ Given the data, divides the histogram edges in such a way that every bin contains at least a
    minimum number of points
    
    Arguments:
        x: [list] Input data.
        points_per_bin: [int] Number of point per bin
    """
    
    nbins = len(x)//points_per_bin
    return histogramEdgesEqualDataNumber(x, nbins)

#########

def rollingAverage2d(x, y, x_window):
    """
    Rolling average where the window is on the x axis rather than index
    
    Arguments:
        x: [list or ndarray] sorted x values
        y: [list or ndarray] y values corresponding to x
        x_window: [float]
        
    Returns:
        output_x: [list]
        output_y: [list]
    """
    assert len(x) == len(y)
    cum_y = np.cumsum(np.insert(y, 0, 0))
    output_y = []
    output_x = []
    j = 0
    for i in range(len(y)):
        for j in range(j, len(y)):
            if x_window >= x[j] - x[i]:
                output_y.append((cum_y[j + 1] - cum_y[i])/(j + 1 - i))
                output_x.append((x[j] + x[i])/2)
            if j > i and (j == len(x) - 1 or (x[j + 1] - x[i] > x_window and x[j] - x[i + 1] < x[j + 1] - x[i])):
                break

    return output_x, output_y



def dimHypot(t1, t2):
    """ 
    Compute the cartesian difference between vectors in an arbitrary number of dimensions

    Arguments:
        t1: [tuple] of a vector
        t2: [tuple] of a vector

    Returns:
        [float] Euclidean difference between vectors or scalars

    """

    t1 = (t1, 0) if type(t1) == int else t1
    t2 = (t2, 0) if type(t2) == int else t2

    if len(t1) != len(t2):
        raise ValueError("Unable to compute cartesian difference between vectors with different dimensions")
        return None

    a = 0
    for c1, c2 in zip(t1, t2):
        a += (c1 - c2) ** 2

    return np.sqrt(a)



##############################################################################################################
# TESTS
# (work in progress)
##############################################################################################################

def testDimHypot():


    test_list = [[1, 2,
                                        1.0],
                 [(1, 1), (4, 5),
                                            5],
                 [(1, 2, 3, 4), (5, 6, 7, 8),
                                            8],
                 [(-1, -2, -3, -4), (-5, -6, -7, -8),
                                            8],
                 [(-1, -2, -3), (-5, -6, -7, -8),
                    "ValueError('Unable to compute cartesian difference between vectors with different dimensions')"]]

    for t1, t2, res in test_list:
        try:
            test_res = dimHypot(t1, t2)
        except Exception as e:
            test_res = repr(e)
        if test_res != res:
            print("Test failure: t1:{}, t2:{}, did not give {}, returned {}"
                                                    .format(t1, t2, res, test_res))


            return False

    return True

def testAngSeparationDeg():


    test_list =     [[[45, 45], [45, 45], 0 ],
                     [[0,  90], [0,   0], 90],
                     [[45, 45], [0,   0], 60],
                     [[30, -30], [-30, 30], 82.8192]]

    for a1, a2, res in test_list:
        if round(angularSeparationDeg(a1[0], a1[1], a2[0], a2[1]), 4) != round(res,4):
            print("Test failure: ra1:{}, dec1:{}, ra2:{}, dec2:{} did not give {}, returned {}"
                            .format(a1[0], a1[1], a2[0], a2[1], res, angularSeparationDeg(a1[0], a1[1], a2[0], a2[1])))
            return False

    return True


def testPointInsideConvexPolygonSphere():

    # # Create a polygon in the sky (has to be convex)
    # perimiter_polygon = [
    #     # RA, Dec
    #     [ 46.00, 79.00],
    #     [136.00, 84.00],
    #     [216.55, 75.82],
    #     [245.85, 61.53],
    #     [319.86, 56.92],
    #     [336.00, 70.64],
    #     [0.00, 77.71],
    #     ]

    perimiter_polygon = [
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
    
    perimiter_polygon = np.array(perimiter_polygon)

    # test_points = [
    #     # RA, Dec
    #     [47.11, 83.83], # Inside
    #     [50.23, 74.42], # Outside
    #     [255.01, 77.33], # Inside
    #     [185.01, 69.45], # Outside
    #     [316.75, 64.33] # Outside
    # ]

    # test_points = np.array(test_points)

    # Sample test points between 0 and 360 degrees in RA and 0 - 90 Declination
    ra_samples = np.linspace(0, 360, 40)
    dec_samples = np.linspace(np.min(perimiter_polygon[:, 1]), 90, 20)
    test_points = np.array(np.meshgrid(ra_samples, dec_samples)).T.reshape(-1, 2)


    # Sort the vertices by RA
    #perimiter_polygon = perimiter_polygon[np.argsort(perimiter_polygon[:, 0])][::-1]

    # # Sort the perimiter by Dec
    # perimiter_polygon = perimiter_polygon[np.argsort(perimiter_polygon[:, 1])]


    # Compute the points that are inside the polygon
    inside = pointInsideConvexPolygonSphere(test_points, perimiter_polygon)

    # Make the plot in polar coordiantes
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Plot the test points
    test_points = np.array(test_points)
    ax.scatter(np.radians(test_points[:, 0]), np.radians(90 - test_points[:, 1]), c='k')

    # Mark the points inside the polygon with an empry green circle
    ax.scatter(np.radians(test_points[inside, 0]), np.radians(90 - test_points[inside, 1]), edgecolors='g', facecolors='none', s=100, label='Inside')

    # Add the first point to close the polygon
    ra_dec = np.vstack((perimiter_polygon, perimiter_polygon[0]))

    # Plot the perimeter of the polygon as a continours curve
    plt.plot(np.radians(ra_dec[:, 0]), np.radians(90 - ra_dec[:, 1]), 'k-')

    # Mark the vertices of the polygon with numbers
    for i in range(len(perimiter_polygon)):
        ax.text(np.radians(perimiter_polygon[i, 0]), np.radians(90 - perimiter_polygon[i, 1]), str(i), fontsize=12)

    plt.legend()
    

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    plt.show()


def tests():

    function_to_test = [["dimHypot", testDimHypot()],
                         ["angSeparationDeg", testAngSeparationDeg()],
                         ["pointInsideConvexPolygonSphere", testPointInsideConvexPolygonSphere()]]

    for func_name, func in function_to_test:
        if func:
            print("Test of {} successful".format(func_name))
        else:
            print("Test failed")



if __name__ == "__main__":

    # Run the tests
    tests()