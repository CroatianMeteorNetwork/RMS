""" Functions for working with AST astrometric plates. """

from __future__ import print_function, division, absolute_import

import os

import numpy as np

from RMS.Astrometry.ApplyAstrometry import calculateMagnitudes
from RMS.Astrometry.Conversions import date2JD, altAz2RADec

class AstPlate(object):
    """ AST type plate structure. """

    def __init__(self):

        self.magic = 0
        self.info_len = 0
        self.star_len = 0
        self.stars = 0

        self.r0 = 0
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0

        self.text = ''
        self.starcat = ''
        self.sitename = ''
        self.site = 0

        self.lat = 0.0
        self.lon = 0.0
        self.elev = 0.0
        self.ts = 0
        self.tu = 0

        self.th0 = 0.0
        self.phi0 = 0.0

        self.rot = 0.0

        self.wid = 0
        self.ht = 0

        self.a = 0
        self.da = 0
        self.b = 0
        self.db = 0
        self.c = 0
        self.dc = 0
        self.d = 0
        self.dd = 0

        self.flags = 0


    def initM(self):
        """ Calculates the conversion matrix. """

        # Init the conversion matrix
        M = np.zeros((3,3))

        M[0,0] = -np.sin(self.phi0)
        M[1,0] =  np.cos(self.phi0)
        M[2,0] =  0.0

        M[0,1] = -np.cos(self.th0)*np.cos(self.phi0)
        M[1,1] = -np.cos(self.th0)*np.sin(self.phi0)
        M[2,1] =  np.sin(self.th0)

        M[0,2] =  np.sin(self.th0)*np.cos(self.phi0)
        M[1,2] =  np.sin(self.th0)*np.sin(self.phi0)
        M[2,2] =  np.cos(self.th0)

        self.M = M

        # Calculate the reverse map matrix
        self.R = np.linalg.inv(M)


    def setFOV(self, azim, elev, wid, ht):
        """ Sets the centre of the FOV with given azimuth and elevation and image size in pixels. 
    
        Arguments:
            azim: [float] Azimuth +E of due N (degrees).
            elev: [float] Elevation angle (degrees).
            wid: [int] Image width (pixels).
            ht: [int] Image height (pixels).

        """

        self.th0 = np.pi/2.0 - np.radians(elev)
        self.phi0 = np.pi/2.0 - np.radians(azim)

        self.wid = wid
        self.ht = ht

        # Init the transformation matrix
        self.initM()


    def __repr__(self):


        return "text " + str(self.text) + " starcat " + str(self.starcat) + " sitename " + \
            str(self.sitename) + " site " + str(self.site) + " lat " + str(self.lat) + " lon " + \
            str(self.lon) + " elev " + str(self.elev) + " ts " + str(self.ts) + " tu " + str(self.tu) + \
            " th0 " + str(np.degrees(self.th0)) + " phi0 " + str(np.degrees(self.phi0)) + " wid " + str(self.wid) + " ht " + \
            str(self.ht) + " a " + str(self.a) + " da " + str(self.da) + " b " + str(self.b) + " db " + \
            str(self.db) + " c " + str(self.c) + " dc " + str(self.dc) + " d " + str(self.d) + " dd " + \
            str(self.dd) + " flags " + str(self.flags) + " rot {:.2f} deg".format(np.degrees(self.rot)) \
            + " FOV: {:.2f} x {:.2f} deg".format(np.degrees(self.fov), np.degrees(self.fov/self.asp))




def loadAST(dir_path, file_name):
    """ Loads an AST plate. 
    
    Arguments:
        dir_path: [str] path to the directory where the plate file is located
        file_name: [str] name of the plate file

    Return:
        ast: [AstPlate object]
    """


    # Open the file for binary reading
    fid = open(os.path.join(dir_path, file_name), 'rb')

    # Init the plate struct
    ast = AstPlate()

    # Load header
    ast.magic = np.fromfile(fid, dtype=np.uint32, count=1)
    ast.info_len = np.fromfile(fid, dtype=np.uint32, count=1)

    # Star structure size in bytes
    ast.star_len = np.fromfile(fid, dtype=np.uint32, count=1)

    # Number of stars
    ast.stars = np.fromfile(fid, dtype=np.int32, count=1)

    # Reserved
    ast.r0 = np.fromfile(fid, dtype=np.int32, count=1)
    ast.r1 = np.fromfile(fid, dtype=np.int32, count=1)
    ast.r2 = np.fromfile(fid, dtype=np.int32, count=1)
    ast.r3 = np.fromfile(fid, dtype=np.int32, count=1)

    # Text description
    ast.text = np.fromfile(fid, dtype='|S'+str(256), count=1)[0].tostring().decode()

    # Name of catalogue
    ast.starcat = np.fromfile(fid, dtype='|S'+str(32), count=1)[0].tostring().decode()

    # Name of observing site
    ast.sitename = np.fromfile(fid, dtype='|S'+str(32), count=1)[0].tostring().decode()

    # Site geo coordinates
    ast.lat = np.fromfile(fid, dtype=np.float64, count=1)[0]
    ast.lon = np.fromfile(fid, dtype=np.float64, count=1)[0]
    ast.elev = np.fromfile(fid, dtype=np.float64, count=1)[0]

    # UNIX time for fit
    ast.ts = np.fromfile(fid, dtype=np.int32, count=1)
    ast.tu = np.fromfile(fid, dtype=np.int32, count=1)

    # Centre of plate
    ast.th0 = np.fromfile(fid, dtype=np.float64, count=1)[0]
    ast.phi0 = np.fromfile(fid, dtype=np.float64, count=1)[0]

    # Original image size
    ast.wid = np.fromfile(fid, dtype=np.int32, count=1)[0]
    ast.ht = np.fromfile(fid, dtype=np.int32, count=1)[0]

    ### Fit parameters
    # x/y --> th
    ast.a = np.fromfile(fid, dtype=np.float64, count=10)
    ast.da = np.fromfile(fid, dtype=np.float64, count=10)

    # x/y --> phi
    ast.b = np.fromfile(fid, dtype=np.float64, count=10)
    ast.db = np.fromfile(fid, dtype=np.float64, count=10)

    # th/phi --> x
    ast.c = np.fromfile(fid, dtype=np.float64, count=10)
    ast.dc = np.fromfile(fid, dtype=np.float64, count=10)

    # th/phi --> y
    ast.d = np.fromfile(fid, dtype=np.float64, count=10)
    ast.dd = np.fromfile(fid, dtype=np.float64, count=10)
    
    # Fit flags
    ast.flags = np.fromfile(fid, dtype=np.uint32, count=1)

    # Calculate the conversion matrix
    ast.initM()


    u = 0.5*ast.wid
    v = 0.5*ast.ht

    ### Calculate the FOV ###

    a, b = plateASTMap(ast, 0.0, v)
    c, d = plateASTMap(ast, 2.0*u, v)

    dot  = np.sin(a)*np.cos(b)*np.sin(c)*np.cos(d)
    dot += np.sin(a)*np.sin(b)*np.sin(c)*np.sin(d)
    dot += np.cos(a)*np.cos(c)

    # FOV width in radians
    ast.fov = np.arccos(dot)

    
    a, b = plateASTMap(ast, u, 0.0)
    c, d = plateASTMap(ast, u, 2.0*v)

    dot  = np.sin(a)*np.cos(b)*np.sin(c)*np.cos(d)
    dot += np.sin(a)*np.sin(b)*np.sin(c)*np.sin(d)
    dot += np.cos(a)*np.cos(c)

    # FOV ascept (width/height)
    ast.asp = ast.fov/np.arccos(dot)

    ######


    ### Calculate the rotation ###
    a, b = plateASTMap(ast, u, v)
    x, y = plateASTMap(ast, a - np.radians(1.0), b, reverse_map=True)

    rot = np.arctan2(v - y, x - u) - np.pi/2
    if rot < -np.pi:
        rot += 2*np.pi

    ast.rot = rot

    ######


    return ast




def plateASTconv(ast, th, phi):
    """ Map theta, phi to gnomonic coordinates. """

    R = ast.R

    # Init vector v
    v = np.zeros(3)

    v[0] = np.sin(th)*np.cos(phi)
    v[1] = np.sin(th)*np.sin(phi)
    v[2] = np.cos(th)

    # Calculate vector u
    u = np.zeros(3)     

    u[0] = R[0,0]*v[0] + R[0,1]*v[1] + R[0,2]*v[2]
    u[1] = R[1,0]*v[0] + R[1,1]*v[1] + R[1,2]*v[2]
    u[2] = R[2,0]*v[0] + R[2,1]*v[1] + R[2,2]*v[2]

    # Calculate beta and gamma
    bet = np.arctan2(np.hypot(u[0], u[1]), u[2])
    gam = np.arctan2(u[1], u[0])

    if bet > np.pi:
        return False

    # Project onto (p, q) plane
    p = np.sin(bet)*np.cos(gam)
    q = np.sin(bet)*np.sin(gam)


    return p, q

# Vectorize the forward mapping function
plateASTconv = np.vectorize(plateASTconv, excluded=['ast'])



def plateASTundo(ast, p, q):
    """ Map gnomonic coordinates to theta, phi. """

    M = ast.M

    # Calculate beta and gamma
    bet = np.arcsin(np.hypot(p, q))
    gam = np.arctan2(q, p)

    if bet > np.pi:
        return False

    # Init vector v
    v = np.zeros(3)

    v[0] = np.sin(bet)*np.cos(gam)
    v[1] = np.sin(bet)*np.sin(gam)
    v[2] = np.cos(bet)

    # Calculate vector u
    u = np.zeros(3)        

    u[0] = M[0,0]*v[0] + M[0,1]*v[1] + M[0,2]*v[2]
    u[1] = M[1,0]*v[0] + M[1,1]*v[1] + M[1,2]*v[2]
    u[2] = M[2,0]*v[0] + M[2,1]*v[1] + M[2,2]*v[2]

    # Convert to theta, phi
    th  = np.arctan2(np.hypot(u[0], u[1]), u[2])
    phi = np.arctan2(u[1], u[0])

    return th, phi

# Vectorize the reverse mapping function
plateASTundo = np.vectorize(plateASTundo, excluded=['ast'])



def plateASTMap(ast, x, y, reverse_map=False):
    """ Map the cartesian (image or galvo encoder) coordinates (x, y) to sky coordinates (theta, phi) given an 
        appropriate ast plate. If a reverse mapping is desired, set reverse_map=True.
        Theta is the zenith angle, phi is the azimuth +N of due E.
        
    Arguments:
        ast: [AstPlate object] AST plate structure.
        x: [float] input parameter 1 (x by default, theta in radians if reverse_map=True).
        y: [float] input parameter 2 (y by default, phi in radians if reverse_map=True).

    Kwargs:
        reverse_map: [bool] Default False, if True, revese mapping is performed.

    Return:
        [tuple of floats]: Output parameters (theta, phi in radians) by default, (x, y) if reverse_map=True.

    """

    # Forward mapping
    if not reverse_map:

        # Normalize coordinates to 0
        x -= ast.wid/2.0
        y -= ast.ht/2.0

        a = ast.a
        b = ast.b

        # Project onto (p, q) plane
        p = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*y + a[5]*y**2 + a[6]*y**3 + a[7]*x*y + a[8]*x**2*y + \
            a[9]*x*y**2

        q = b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3 + b[4]*y + b[5]*y**2 + b[6]*y**3 + b[7]*x*y + b[8]*x**2*y + \
            b[9]*x*y**2

        
        # Map gonomnic coordinates to theta, phi
        th, phi = plateASTundo(ast, p, q)

        return th, phi


    # Reverse mapping
    else:

        th, phi = x, y
            
        c = ast.c
        d = ast.d

        # Map theta, phi to gnomonic coordinates
        p, q = plateASTconv(ast, th, phi)

        u = c[0] + c[1]*p + c[2]*p**2 + c[3]*p**3 + c[4]*q + c[5]*q**2 + c[6]*q**3 + c[7]*p*q + c[8]*p**2*q + \
            c[9]*p*q**2

        v = d[0] + d[1]*p + d[2]*p**2 + d[3]*p**3 + d[4]*q + d[5]*q**2 + d[6]*q**3 + d[7]*p*q + d[8]*p**2*q + \
            d[9]*p*q**2

        # Calculate Hx, Hy
        x = u + ast.wid/2.0
        y = v + ast.ht/2.0

        return x, y



def xyToRaDecAST(time_data, X_data, Y_data, level_data, ast, photom_offset):
    """ Converts image XY to RA,Dec, but it takes a platepar instead of individual parameters. 
    
    Arguments:
        time_data: [2D ndarray] Numpy array containing time tuples of each data point (year, month, day, 
            hour, minute, second, millisecond).
        X_data: [ndarray] 1D numpy array containing the image X component.
        Y_data: [ndarray] 1D numpy array containing the image Y component.
        level_data: [ndarray] Levels of the meteor centroid.
        ast: [AstPlate object] AST plate structure.
        photom_offset: [float] Photometric offset used to compute the magnitude.


    Return:
        (JD_data, RA_data, dec_data, magnitude_data): [tuple of ndarrays]
            JD_data: [ndarray] Julian date of each data point.
            RA_data: [ndarray] Right ascension of each point (deg).
            dec_data: [ndarray] Declination of each point (deg).
            magnitude_data: [ndarray] Array of meteor's lightcurve apparent magnitudes.
    """

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # Compute theta and phi from X, Y
    theta_data, phi_data = plateASTMap(ast, X_data, Y_data)

    # Compute altitude and azimuth in degrees
    azimuth_data = np.degrees(np.pi/2 - phi_data)
    altitude_data = np.degrees(np.pi/2 - theta_data)

    # Convert azimuth (+E of due N) and altitude to RA and Dec
    JD_data = [date2JD(*t) for t in time_data]
    RA_data, dec_data = altAz2RADec(azimuth_data, altitude_data, JD_data, np.degrees(ast.lat), \
        np.degrees(ast.lon))


    # Calculate magnitudes (ignore vignetting)
    magnitude_data = calculateMagnitudes(level_data, np.zeros_like(level_data), photom_offset, 0.0)


    return JD_data, RA_data, dec_data, magnitude_data

