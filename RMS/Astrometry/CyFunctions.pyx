import numpy as np
# import cv2

# Cython import
cimport numpy as np
cimport cython

# Define numpy types
INT_TYPE = np.uint32
ctypedef np.uint32_t INT_TYPE_t

FLOAT_TYPE = np.float64 
ctypedef np.float64_t FLOAT_TYPE_t


# Define Pi
cdef double pi = np.pi

# Declare math functions
cdef extern from "math.h":
    double sin(double)
    double asin(double)
    double cos(double)
    double acos(double)
    double atan2(double, double)
    double sqrt(double)


@cython.cdivision(True)
cdef double radians(double deg):
    """Converts degrees to radians.
    """
    
    return deg/180.0*(pi)

@cython.cdivision(True)
cdef double degrees(double deg):
    """Converts radians to degrees.
    """
    
    return deg*180.0/pi

@cython.boundscheck(False)
@cython.wraparound(False) 
cpdef double angularSeparation(double ra1, double dec1, double ra2, double dec2):
    """ Calculate the angular separation between 2 stars in equatorial celestial coordinates. 

    Source of the equation: http://www.astronomycafe.net/qadir/q1890.html (May 1, 2016)

    @param ra1: [float] right ascension of the first stars (in degrees)
    @param dec1: [float] decliantion of the first star (in degrees)
    @param ra2: [float] right ascension of the decons stars (in degrees)
    @param dec2: [float] decliantion of the decons star (in degrees)

    @return angular_separation: [float] angular separation (in degrees)
    """

    # Convert input coordinates to radians
    ra1 = radians(ra1)
    dec1 =  radians(dec1)
    ra2 = radians(ra2)
    dec2 = radians(dec2)

    return degrees(acos(sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra2 - ra1)))



@cython.boundscheck(False)
@cython.wraparound(False) 
def subsetCatalog(np.ndarray[FLOAT_TYPE_t, ndim=2] catalog_list, double ra_c, double dec_c, double radius, \
        double mag_limit):
    """ Make a subset of stars from the given star catalog around the given coordinates with a given radius.
    
    Arguments:
        ...
        ra_c: [float] Centre of extraction RA (degrees).
        dec_c: [float] Centre of extraction dec (degrees).
        radius: [float] Extraction radius (degrees).
        ...

    """


    # Define variables
    cdef int i, k
    cdef double dec_min, dec_max
    cdef double ra, dec, mag
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] filtered_list = np.zeros(shape=(catalog_list.shape[0], \
        catalog_list.shape[1]), dtype=FLOAT_TYPE)

    cdef np.ndarray[INT_TYPE_t, ndim=1] filtered_indices = np.zeros(shape=(catalog_list.shape[0]), \
        dtype=INT_TYPE)

    # Calculate minimum and maximum declination
    dec_min = dec_c - radius
    if dec_min < -90:
        dec_min = -90

    dec_max = dec_c + radius
    if dec_max > 90:
        dec_max = 90

    k = 0
    for i in range(catalog_list.shape[0]):

        ra = catalog_list[i,0]
        dec = catalog_list[i,1]
        mag = catalog_list[i,2]

        # Skip if the declination is too large
        if dec > dec_max:
            continue

        # End the loop if the declination is too small
        if dec < dec_min:
            break

        # Add star to the list if it is within a given radius and has a certain brightness
        if (angularSeparation(ra, dec, ra_c, dec_c) <= radius) and (mag <= mag_limit):
            
            filtered_list[k,0] = ra
            filtered_list[k,1] = dec
            filtered_list[k,2] = mag

            # Add index to the list of indices which passed the filter
            filtered_indices[k] = i;

            # Increment filtered list counter
            k += 1


    return filtered_indices[:k], filtered_list[:k]



@cython.boundscheck(False)
@cython.wraparound(False)
def matchStars(np.ndarray[FLOAT_TYPE_t, ndim=2] stars_list, np.ndarray[FLOAT_TYPE_t, ndim=1] cat_x_array, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] cat_y_array, np.ndarray[INT_TYPE_t, ndim=1] cat_good_indices, \
    double max_radius):
        

    cdef int i, j
    cdef unsigned int cat_idx
    cdef int k = 0
    cdef double min_dist, dist
    cdef double cat_match_indx, im_star_y, im_star_x, cat_x, cat_y

    # Get the lenghts of input arrays
    cdef int stars_len = stars_list.shape[0]
    cdef int cat_len = cat_good_indices.shape[0]
    
    # List for matched indices
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] matched_indices = np.zeros(shape=(stars_list.shape[0], 3), \
        dtype=FLOAT_TYPE)

    
    ### Match image and catalog stars ###

    # Go through all image stars
    for i in range(stars_len):

        # Extract image star coordinates
        im_star_y = stars_list[i, 0]
        im_star_x = stars_list[i, 1]

        min_dist = max_radius
        cat_match_indx = -1

        # Check for the best match among catalog stars
        for j in range(cat_len):

            cat_idx = cat_good_indices[j]

            # Extract catalog coordinates
            cat_x = cat_x_array[cat_idx]
            cat_y = cat_y_array[cat_idx]


            # Calculate the distance between stars
            dist = sqrt((im_star_x - cat_x)**2 + (im_star_y - cat_y)**2)

            # Set the catalog star as the best match if it is the closest to the image star than any previous
            if (dist < min_dist):
                min_dist = dist
                cat_match_indx = cat_idx


        # Take the best matched star if the distance was within the maximum radius
        if min_dist < max_radius:
            
            # Add the matched indices to the output list
            matched_indices[k, 0] = i
            matched_indices[k, 1] = cat_match_indx
            matched_indices[k, 2] = min_dist

            k += 1


    
    # Cut the output list to the number of matched stars
    matched_indices = matched_indices[:k]

    return matched_indices



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyraDecToXY(np.ndarray[FLOAT_TYPE_t, ndim=1] RA_data, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data, double jd, double lat, double lon, double x_res, \
    double y_res, double az_centre, double alt_centre, double pos_angle_ref, double F_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_rev, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_rev):
    """ Convert RA, Dec to distorion corrected image coordinates. 

    Arguments:
        RA: [ndarray] Array of right ascensions (degrees).
        dec: [ndarray] Array of declinations (degrees).
        jd: [float] Julian date.
        lat: [float] Latitude of station in degrees.
        lon: [float] Longitude of station in degrees.
        x_res: [int] X resolution of the camera.
        y_res: [int] Y resolution of the camera.
        az_centre: [float] Azimuth of the FOV centre (degrees).
        alt_centre: [float] Altitude of the FOV centre (degrees).
        pos_angle_ref: [float] Rotation from the celestial meridial (degrees).
        F_scale: [float] Image scale (px/deg).
        x_poly_rev: [ndarray float] Distorsion polynomial in X direction for reverse mapping.
        y_poly_rev: [ndarray float] Distorsion polynomail in Y direction for reverse mapping.
    
    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.
    """

    cdef int i
    cdef double ra_star, dec_star
    cdef double ra1, dec1, ra2, dec2, ad, radius, sinA, cosA, theta, X1, Y1, dX, dY

    # print('jd:', jd)

    # Calculate the reference hour angle
    cdef double T = (jd - 2451545.0)/36525.0
    cdef double Ho = (280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T**2 \
        - (T**3)/38710000.0)%360

    cdef double sl = sin(radians(lat))
    cdef double cl = cos(radians(lat))

    # Calculate the hour angle
    cdef double salt = sin(radians(alt_centre))
    cdef double saz = sin(radians(az_centre))
    cdef double calt = cos(radians(alt_centre))
    cdef double caz = cos(radians(az_centre))
    cdef double x = -saz*calt
    cdef double y = -caz*sl*calt + salt*cl
    cdef double HA = degrees(atan2(x, y))

    # Centre of FOV
    cdef double RA_centre = (Ho + lon - HA)%360
    cdef double dec_centre = degrees(asin(sl*salt + cl*calt*caz))

    # print('RA centre:', RA_centre)
    # print('Dec centre:', dec_centre)

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array = np.zeros_like(RA_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] y_array = np.zeros_like(RA_data)

    for i in range(RA_data.shape[0]):

        ra_star = RA_data[i]
        dec_star = dec_data[i]

        # Gnomonization of star coordinates to image coordinates
        ra1 = radians(RA_centre)
        dec1 = radians(dec_centre)
        ra2 = radians(ra_star)
        dec2 = radians(dec_star)
        ad = acos(sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra2 - ra1))
        radius = degrees(ad)
        sinA = cos(dec2)*sin(ra2 - ra1)/sin(ad)
        cosA = (sin(dec2) - sin(dec1)*cos(ad))/(cos(dec1)*sin(ad))
        theta = -degrees(atan2(sinA, cosA))
        theta = theta + pos_angle_ref - 90.0

        #dist = np.degrees(acos(sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra1 - ra2)))

        # Calculate the image coordinates
        X1 = radius*cos(radians(theta))*F_scale
        Y1 = radius*sin(radians(theta))*F_scale

        # Calculate distortion in X direction
        dX = (x_poly_rev[0]
            + x_poly_rev[1]*X1
            + x_poly_rev[2]*Y1
            + x_poly_rev[3]*X1**2
            + x_poly_rev[4]*X1*Y1
            + x_poly_rev[5]*Y1**2
            + x_poly_rev[6]*X1**3
            + x_poly_rev[7]*X1**2*Y1
            + x_poly_rev[8]*X1*Y1**2
            + x_poly_rev[9]*Y1**3
            + x_poly_rev[10]*X1*sqrt(X1**2 + Y1**2)
            + x_poly_rev[11]*Y1*sqrt(X1**2 + Y1**2))

        # Add the distortion correction and calculate X image coordinates
        x_array[i] = X1 - dX + x_res/2.0

        # Calculate distortion in Y direction
        dY = (y_poly_rev[0]
            + y_poly_rev[1]*X1
            + y_poly_rev[2]*Y1
            + y_poly_rev[3]*X1**2
            + y_poly_rev[4]*X1*Y1
            + y_poly_rev[5]*Y1**2
            + y_poly_rev[6]*X1**3
            + y_poly_rev[7]*X1**2*Y1
            + y_poly_rev[8]*X1*Y1**2
            + y_poly_rev[9]*Y1**3
            + y_poly_rev[10]*Y1*sqrt(X1**2 + Y1**2)
            + y_poly_rev[11]*X1*sqrt(X1**2 + Y1**2))

        # Add the distortion correction and calculate Y image coordinates
        y_array[i] = Y1 - dY + y_res/2.0


    return x_array, y_array



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyXYToRADec(np.ndarray[FLOAT_TYPE_t, ndim=1] jd_data, np.ndarray[FLOAT_TYPE_t, ndim=1] X_data, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] Y_data, double lat, double lon, double Ho, double X_res, double Y_res, \
    double RA_d, double dec_d, double pos_angle_ref, double F_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd):
    """
    Arguments:
        jd_data: [ndarray] Julian date of each data point.
        X_data: [ndarray] 1D numpy array containing the image column.
        Y_data: [ndarray] 1D numpy array containing the image row.
        lat: [float] Latitude of the observer in degrees.
        lon: [float] Longitde of the observer in degress.
        Ho: [float] Reference hour angle (deg).
        X_res: [int] Image size, X dimension (px).
        Y_res: [int] Image size, Y dimenstion (px).
        RA_d: [float] Reference right ascension of the image centre (degrees).
        dec_d: [float] Reference declination of the image centre (degrees).
        pos_angle_ref: [float] Field rotation parameter (degrees).
        F_scale: [float] Sum of image scales per each image axis (arcsec per px).
        x_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward X axis polynomial parameters.
        y_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward Y axis polynomial parameters.
    
    Return:
        (RA_data, dec_data): [tuple of ndarrays]
            
            RA_data: [ndarray] Right ascension of each point (deg).
            dec_data: [ndarray] Declination of each point (deg).
            magnitude_data: [ndarray] Array of meteor's lightcurve apparent magnitudes.
    """

    cdef int i
    cdef double jd, x_det, y_det, dx, x_pix, dy, y_pix
    cdef double dec_rad, sl, cl
    cdef double radius, theta, sin_t, Dec0det, cos_t, RA0det, h, sh, sd, ch, cd, x, y, z, r, azimuth, altitude
    cdef double az_rad, alt_rad, saz, salt, caz, calt, HA, T, RA, dec, hour_angle

    # Convert declination to radians
    dec_rad = radians(dec_d)

    # Precalculate some parameters
    sl = sin(radians(lat))
    cl = cos(radians(lat))

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] RA_data = np.zeros_like(jd_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data = np.zeros_like(jd_data)

    # Go through all given data points
    for i in range(jd_data.shape[0]):

        jd = jd_data[i]
        x_det = X_data[i]
        y_det = Y_data[i]


        ### APPLY DISTORSION CORRECTION ###

        x_det = x_det - X_res/2.0
        y_det = y_det - Y_res/2.0

        dx = (x_poly_fwd[0]
            + x_poly_fwd[1]*x_det
            + x_poly_fwd[2]*y_det
            + x_poly_fwd[3]*x_det**2
            + x_poly_fwd[4]*x_det*y_det
            + x_poly_fwd[5]*y_det**2
            + x_poly_fwd[6]*x_det**3
            + x_poly_fwd[7]*x_det**2*y_det
            + x_poly_fwd[8]*x_det*y_det**2
            + x_poly_fwd[9]*y_det**3
            + x_poly_fwd[10]*x_det*sqrt(x_det**2 + y_det**2)
            + x_poly_fwd[11]*y_det*sqrt(x_det**2 + y_det**2))

        # Add the distortion correction
        x_pix = x_det + dx

        dy = (y_poly_fwd[0]
            + y_poly_fwd[1]*x_det
            + y_poly_fwd[2]*y_det
            + y_poly_fwd[3]*x_det**2
            + y_poly_fwd[4]*x_det*y_det
            + y_poly_fwd[5]*y_det**2
            + y_poly_fwd[6]*x_det**3
            + y_poly_fwd[7]*x_det**2*y_det
            + y_poly_fwd[8]*x_det*y_det**2
            + y_poly_fwd[9]*y_det**3
            + y_poly_fwd[10]*y_det*sqrt(x_det**2 + y_det**2)
            + y_poly_fwd[11]*x_det*sqrt(x_det**2 + y_det**2))

        # Add the distortion correction
        y_pix = y_det + dy

        # Scale back image coordinates
        x_pix = x_pix/F_scale
        y_pix = y_pix/F_scale

        ### ###


        ### Convert gnomonic X, Y to alt, az ###

        # Caulucate the needed parameters
        radius = radians(sqrt(x_pix**2 + y_pix**2))
        theta = radians((90 - pos_angle_ref + degrees(atan2(y_pix, x_pix)))%360)

        sin_t = sin(dec_rad)*cos(radius) + cos(dec_rad)*sin(radius)*cos(theta)
        Dec0det = atan2(sin_t, sqrt(1 - sin_t**2))

        sin_t = sin(theta)*sin(radius)/cos(Dec0det)
        cos_t = (cos(radius) - sin(Dec0det)*sin(dec_rad))/(cos(Dec0det)*cos(dec_rad))
        RA0det = (RA_d - degrees(atan2(sin_t, cos_t)))%360

        h = radians(Ho + lon - RA0det)
        sh = sin(h)
        sd = sin(Dec0det)
        ch = cos(h)
        cd = cos(Dec0det)

        x = -ch*cd*sl + sd*cl
        y = -sh*cd
        z = ch*cd*cl + sd*sl

        r = sqrt(x**2 + y**2)

        # Calculate azimuth and altitude
        azimuth = degrees(atan2(y, x))%360
        altitude = degrees(atan2(z, r))

        ### ###


        ### Convert alt, az to RA, Dec ###

        # Never allow the altitude to be exactly 90 deg due to numerical issues
        if altitude == 90:
            altitude = 89.9999

        # Convert altitude and azimuth to radians
        az_rad = radians(azimuth)
        alt_rad = radians(altitude)

        saz = sin(az_rad)
        salt = sin(alt_rad)
        caz = cos(az_rad)
        calt = cos(alt_rad)

        x = -saz*calt
        y = -caz*sl*calt + salt*cl
        HA = degrees(atan2(x, y))

        # Calculate the hour angle
        T = (jd - 2451545.0)/36525.0
        hour_angle = (280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T**2 - T**3/38710000.0)%360

        RA = (hour_angle + lon - HA)%360
        dec = degrees(asin(sl*salt + cl*calt*caz))

        ### ###


        RA_data[i] = RA
        dec_data[i] = dec


    return RA_data, dec_data