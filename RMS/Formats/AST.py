""" Functions for working with AST astrometric plates. """

from __future__ import print_function, division, absolute_import

import os

import numpy as np

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
        [AstPlate object]
    """


    # Open the file for binary reading
    fid = open(os.path.join(dir_path, file_name), 'rb')

    # Init the plate struct
    exact = AstPlate()

    # Load header
    exact.magic = np.fromfile(fid, dtype=np.uint32, count = 1)
    exact.info_len = np.fromfile(fid, dtype=np.uint32, count = 1)

    # Star structure size in bytes
    exact.star_len = np.fromfile(fid, dtype=np.uint32, count = 1)

    # Number of stars
    exact.stars = np.fromfile(fid, dtype=np.int32, count = 1)

    # Reserved
    exact.r0 = np.fromfile(fid, dtype=np.int32, count = 1)
    exact.r1 = np.fromfile(fid, dtype=np.int32, count = 1)
    exact.r2 = np.fromfile(fid, dtype=np.int32, count = 1)
    exact.r3 = np.fromfile(fid, dtype=np.int32, count = 1)

    # Text description
    exact.text = np.fromfile(fid, dtype='|S'+str(256), count = 1)[0]

    # Name of catalogue
    exact.starcat = np.fromfile(fid, dtype='|S'+str(32), count = 1)[0]

    # Name of observing site
    exact.sitename = np.fromfile(fid, dtype='|S'+str(32), count = 1)[0]

    # Site geo coordinates
    exact.lat = np.fromfile(fid, dtype=np.float64, count = 1)[0]
    exact.lon = np.fromfile(fid, dtype=np.float64, count = 1)[0]
    exact.elev = np.fromfile(fid, dtype=np.float64, count = 1)[0]

    # UNIX time for fit
    exact.ts = np.fromfile(fid, dtype=np.int32, count = 1)
    exact.tu = np.fromfile(fid, dtype=np.int32, count = 1)

    # Centre of plate
    exact.th0 = np.fromfile(fid, dtype=np.float64, count = 1)[0]
    exact.phi0 = np.fromfile(fid, dtype=np.float64, count = 1)[0]

    # Original image size
    exact.wid = np.fromfile(fid, dtype=np.int32, count = 1)[0]
    exact.ht = np.fromfile(fid, dtype=np.int32, count = 1)[0]

    ### Fit parameters
    # x/y --> th
    exact.a = np.fromfile(fid, dtype=np.float64, count = 10)
    exact.da = np.fromfile(fid, dtype=np.float64, count = 10)

    # x/y --> phi
    exact.b = np.fromfile(fid, dtype=np.float64, count = 10)
    exact.db = np.fromfile(fid, dtype=np.float64, count = 10)

    # th/phi --> x
    exact.c = np.fromfile(fid, dtype=np.float64, count = 10)
    exact.dc = np.fromfile(fid, dtype=np.float64, count = 10)

    # th/phi --> y
    exact.d = np.fromfile(fid, dtype=np.float64, count = 10)
    exact.dd = np.fromfile(fid, dtype=np.float64, count = 10)
    
    # Fit flags
    exact.flags = np.fromfile(fid, dtype=np.uint32, count = 1)

    # Calculate the conversion matrix
    exact.initM()


    u = 0.5*exact.wid
    v = 0.5*exact.ht

    ### Calculate the FOV ###

    a, b = plateASTMap(exact, 0.0, v)
    c, d = plateASTMap(exact, 2.0*u, v)

    dot  = np.sin(a)*np.cos(b)*np.sin(c)*np.cos(d)
    dot += np.sin(a)*np.sin(b)*np.sin(c)*np.sin(d)
    dot += np.cos(a)*np.cos(c)

    # FOV width in radians
    exact.fov = np.arccos(dot)

    
    a, b = plateASTMap(exact, u, 0.0)
    c, d = plateASTMap(exact, u, 2.0*v)

    dot  = np.sin(a)*np.cos(b)*np.sin(c)*np.cos(d)
    dot += np.sin(a)*np.sin(b)*np.sin(c)*np.sin(d)
    dot += np.cos(a)*np.cos(c)

    # FOV ascept (width/height)
    exact.asp = exact.fov/np.arccos(dot)

    ######


    ### Calculate the rotation ###
    a, b = plateASTMap(exact, u, v)
    x, y = plateASTMap(exact, a - np.radians(1.0), b, reverse_map=True)

    rot = np.arctan2(v - y, x - u) - np.pi/2
    if rot < -np.pi:
        rot += 2*np.pi

    exact.rot = rot

    ######


    return exact




def plateASTconv(exact, th, phi):
    """ Map theta, phi to gnomonic coordinates. """

    R = exact.R

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



def plateASTundo(exact, p, q):
    """ Map gnomonic coordinates to theta, phi. """

    M = exact.M

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



def plateASTMap(exact, x, y, reverse_map=False):
    """ Map the mirror encoder coordinates (Hx, Hy) to sky coordinates (theta, phi) given an 
        appropriate exact plate. If a reverse mapping is desired, set reverse_map=True.
        
    Arguments:
        scale: [AstPlate object] AST plate structure
        x: [float] input parameter 1 (Hx by default, theta if reverse_map=True)
        y: [float] input parameter 2 (Hy by default, phi if reverse_map=True)

    Kwargs:
        reverse_map: [bool] default False, if True, revese mapping is performed

    Return:
        [tuple of floats]: output parameters (theta, phi) by default, (Hx, Hy) if reverse_map=True

    """

    # Forward mapping
    if not reverse_map:

        # Normalize coordinates to 0
        x -= exact.wid/2.0
        y -= exact.ht/2.0

        a = exact.a
        b = exact.b

        # Project onto (p, q) plane
        p = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*y + a[5]*y**2 + a[6]*y**3 + a[7]*x*y + a[8]*x**2*y + \
            a[9]*x*y**2

        q = b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3 + b[4]*y + b[5]*y**2 + b[6]*y**3 + b[7]*x*y + b[8]*x**2*y + \
            b[9]*x*y**2

        
        # Map gonomnic coordinates to theta, phi
        th, phi = plateASTundo(exact, p, q)

        return th, phi


    # Reverse mapping
    else:

        th, phi = x, y
            
        c = exact.c
        d = exact.d

        # Map theta, phi to gnomonic coordinates
        p, q = plateASTconv(exact, th, phi)

        u = c[0] + c[1]*p + c[2]*p**2 + c[3]*p**3 + c[4]*q + c[5]*q**2 + c[6]*q**3 + c[7]*p*q + c[8]*p**2*q + \
            c[9]*p*q**2

        v = d[0] + d[1]*p + d[2]*p**2 + d[3]*p**3 + d[4]*q + d[5]*q**2 + d[6]*q**3 + d[7]*p*q + d[8]*p**2*q + \
            d[9]*p*q**2

        # Calculate Hx, Hy
        x = u + exact.wid/2.0
        y = v + exact.ht/2.0

        return x, y