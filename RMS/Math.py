""" Common math functions. """

import numpy as np



def lineFunc(x, m, k):
    """ Linear function.
    
    Arguments:
        x: [float or ndarray] Independant variable.
        m: [float] Slope.
        k: [float] Y-intercept.

    Return:
        y: [float or ndarray] Dependant variable.
    """

    return m*x + k


def logLineFunc(x, m, k):
    """ Logarithmic linear function.

    Arguments:
        x: [float or ndarray] Independant variable.
        m: [float] Slope.
        k: [float] Y-intercept.

    Return:
        y: [float or ndarray] Dependant variable.
    """

    return 10**lineFunc(x, m, k)


def angularSeparation(ra1, dec1, ra2, dec2):
    """ Calculates the angle between two points on a sphere. 
    
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
        (x, y, z): [tuple of floats] Coordinates of the point in 3D cartiesian coordinates.
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
        (x, y, z): [tuple of floats] Coordinates of the point in 3D cartiesian coordinates.
    """

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return x, y, z


def pointInsideConvexPolygonSphere(points, vertices):
    """
    Polygon must be convex
    https://math.stackexchange.com/questions/4012834/checking-that-a-point-is-in-a-spherical-polygon


    Arguments:
        points: [array] points with dimension (npoints, 2). The two dimensions are ra and dec
        vertices: [array] vertices of convex polygon with dimension (nvertices, 2)
        
    Return:
        filter: [array of bool] Array of booleans on whether a given point is inside the polygon on 
            the sphere.
    """
    # convert ra dec to spherical
    points = points[:, ::-1]
    vertices = vertices[:, ::-1]
    points[:, 0] = 90 - points[:, 0]
    vertices[:, 0] = 90 - vertices[:, 0]
    points = np.array(sphericalToCartesian(*np.hstack((np.ones((len(points), 1)), np.radians(points))).T))
    vertices = np.array(sphericalToCartesian(*np.hstack((np.ones((len(vertices), 1)), np.radians(vertices))).T))
    
    great_circle_normal = np.cross(vertices, np.roll(vertices, 1, axis=1), axis=0)
    dot_prod = np.dot(great_circle_normal.T, points)
    return np.sum(dot_prod < 0, axis=0, dtype=int) == 0  # inside if n . p < 0 for no n


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
