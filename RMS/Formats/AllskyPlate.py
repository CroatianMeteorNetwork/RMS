from __future__ import print_function, division, absolute_import

import numpy as np



class AllskyPlate(object):
    def __init__(self):
        """ Borovicka et al. (1995) all-sky calibration, implementation based on code by Rob Weryk. """

        ### Plate constants

        # Angle between the X axis and the direction to the south
        self.a0 = 0 

        # Coordinates of centre of projection
        self.x0 = self.y0 = 0

        ###

        # Camera constants
        self.A = self.F = self.C = 0
        
        # Lens constants
        self.V = self.S = self.D = self.P = self.Q = 0

        # Station constants
        self.eps = self.E = 0

        # Image geometry
        self.wid = self.ht = 0



    def xy2AltAz(self, px, py):
        """ Maps image coordinates to azimuth and altitude. 
        
        Arguments:
            px: [float] X coordinate.
            py: [float] Y coordinate.

        Return:
            azimuth: [float] Azimuth +E of due N (degrees).
            altitude: [float] Altitude (degrees).
        """

        px = px - self.wid/2.0
        py = self.ht/2.0 - py

        dx0 = px - self.x0
        dy0 = py - self.y0
        da0 = self.F - self.a0

        # Radial distance from centre
        r = self.C*(np.hypot(dx0, dy0) + self.A*dy0*np.cos(da0) - self.A*dx0*np.sin(da0))

        # Angular distance from the centre of projection
        u = self.V*r + self.S*(np.exp(self.D*r) - 1.0) + self.P*(np.exp(self.Q*r**2) - 1.0)

        # Azimuth of projection
        b = self.a0 - self.E + np.arctan2(dy0, dx0)
        b = b%(2*np.pi) - np.pi

        cz = np.cos(u)*np.cos(self.eps) - np.sin(u)*np.sin(self.eps)*np.cos(b)
        sz = np.sqrt(1.0 - cz*cz)

        # Compute zenith distance
        z = np.arctan2(sz, cz)

        # Compute altitude
        altitude = np.pi/2.0 - z

        sa = np.sin(b)*np.sin(u)/sz
        ca = (np.cos(u) - np.cos(self.eps)*cz)/np.sin(self.eps)/sz

        # Compute azimuth from due North
        azimuth = self.E + np.arctan2(sa, ca)

        # Wrap azimuth to (0, 2pi) range
        azimuth = azimuth%(2*np.pi)

        azimuth += np.pi

        # Wrap azimuth to (0, 2pi) range
        azimuth = azimuth%(2*np.pi)

        return azimuth, altitude



    def AltAz2xy(self, azimuth, altitude):

        # Compute zenith angle
        z = np.pi/2.0 - altitude

        # Compute azimuth from due South
        a = azimuth - np.pi
        a = a%(2*np.pi)
        
        cu = np.cos(a - self.E)*np.sin(self.eps)*np.sin(z) + np.cos(self.eps)*np.cos(z)
        u = np.arccos(cu)

        sb = np.sin(z)*np.sin(a - self.E)*np.sin(self.eps)
        cb = cu*np.cos(self.eps) - np.cos(z)
        b = np.arctan2(sb, cb)
        da0 = self.F - self.a0

        # Find r using an iterative procedure
        niter = 0
        r = 0.0

        val  = self.V*r + self.S*(np.exp(self.D*r) - 1.0)
        val += self.P*(np.exp(self.Q*r*r) - 1.0)
        res  = u - val

        while (np.abs(res) > 1.0e-8):

            DUR  = self.V + self.S*self.D*np.exp(self.D*r)
            DUR += 2.0*self.P*self.Q*r*np.exp(self.Q*r*r)

            # r is stable to large steps
            r += 0.95*res/DUR

            val  = self.V*r + self.S*(np.exp(self.D*r) - 1.0)
            val += self.P*(np.exp(self.Q*r**2) - 1.0)
            res  = u - val

            if niter > 5000:
                return -1

            niter += 1
        

        #### r IS GOOD AND VERY PRECISLTY INVERTED!!!        

        # Initial guess of coordinates - the direction if OK, but the distance from the centre is not
        dx0 = r*np.cos(b - self.a0 + self.E)
        dy0 = r*np.sin(b - self.a0 + self.E)

        niter = 0
        r_est = self.C*(np.hypot(dx0, dy0) + self.A*dy0*np.cos(da0) - self.A*dx0*np.sin(da0))
        res = r - r_est

        # Estimate the distance from centre until satisfactory
        while (np.abs(res) > 1.0e-4):

            DRX = self.C*(dx0/np.hypot(dx0, dy0) - self.A*np.sin(da0))
            DRY = self.C*(dy0/np.hypot(dx0, dy0) + self.A*np.cos(da0))

            # dx0/dy0 must be stepped slowly to avoid oscillations 
            # dx0 and dy0 are related, so constrain one based on the other

            if (np.abs(DRX) > np.abs(DRY)):
                dx0 += 0.65*res/DRX
                dy0  = dx0*np.tan(b - self.a0 + self.E)
            else:
                dy0 += 0.65*res/DRY
                dx0  = dy0/np.tan(b - self.a0 + self.E)

            r_est = self.C*(np.hypot(dx0, dy0) + self.A*dy0*np.cos(da0) - self.A*dx0*np.sin(da0))
            res = r - r_est

            if niter > 5000:
                return -1

            niter += 1


        tx = self.x0 + dx0
        ty = self.y0 + dy0

        tx += self.wid/2.0
        ty  = self.ht/2.0 - ty


        return tx, ty





if __name__ == "__main__":


    ### Test allsky-calibration functions

    # Init the plate
    plate = AllskyPlate()

    # Plate constants
    plate.a0 = np.radians(69.795406)
    plate.x0 = -28.371173
    plate.y0 = 21.716295

    # Camera constants
    plate.A = 0.000252
    plate.F = np.degrees(-24.203551)
    plate.C = 1.0
    
    # Lens constants
    plate.V = 0.004717
    plate.S = 0.004192
    plate.D = 0.011161
    plate.P = -0.000042
    plate.Q = -0.000667

    # Station constants
    plate.eps = np.degrees(1.424254)
    plate.E = np.degrees(116.561332)

    # Image geometry
    plate.wid = 640
    plate.ht = 480

    # Reference Julian date
    plate.JD = 0


    ########################

    # Input coordinates
    x = 100.0
    y = 100.0

    print(x, y)

    # Forward map test coordinates
    azimuth, altitude = plate.xy2AltAz(x, y)
    print(azimuth - np.pi, np.pi/2 - altitude)

    print("Azim, alt:", np.degrees(azimuth), np.degrees(altitude))

    # Reverse map test
    x_rev, y_rev = plate.AltAz2xy(azimuth, altitude)

    print(x_rev, y_rev)