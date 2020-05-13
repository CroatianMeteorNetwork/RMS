#!python
#cython: language_level=3

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
    double tan(double)
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



@cython.cdivision(True)
cdef double cyjd2LST(double jd, double lon):
    """ Convert Julian date to apparent Local Sidereal Time. The times is apparent, not mean!

    Source: J. Meeus: Astronomical Algorithms

    Arguments:
        jd: [float] Decimal julian date, epoch J2000.0.
        lon: [float] Longitude of the observer in degrees.
    
    Return:
        lst [float] Apparent Local Sidereal Time (deg).
    """

    cdef double gst

    # Define the Julian date at the J2000 epoch
    cdef double J2000_days = 2451545.0



    cdef double t = (jd - J2000_days)/36525.0

    # Calculate the Mean sidereal rotation of the Earth in radians (Greenwich Sidereal Time)
    gst = 280.46061837 + 360.98564736629*(jd - J2000_days) + 0.000387933*t**2 - (t**3)/38710000.0
    gst = (gst + 360)%360


    # Compute the apparent Local Sidereal Time (LST)
    return (gst + lon + 360)%360



@cython.cdivision(True)
cpdef tuple cyraDec2AltAz(double ra, double dec, double jd, double lat, double lon):
    """ Convert right ascension and declination to azimuth (+east of sue north) and altitude. 

    Arguments:
        ra: [float] Right ascension in radians.
        dec: [float] Declination in radians.
        jd: [float] Julian date.
        lat: [float] latitude in radians.
        lon: [float] longitude in radians.

    Return:
        (azim, elev): [tuple]
            azim: [float] Azimuth (+east of due north) in radians.
            elev: [float] Elevation above horizon in radians.

        """

    cdef double lst, ha, azim, sin_elev, elev

    # Calculate Local Sidereal Time
    lst = radians(cyjd2LST(jd, degrees(lon)))

    # Calculate the hour angle
    ha = lst - ra

    # Constrain the hour angle to [-pi, pi] range
    ha = (ha + pi)%(2*pi) - pi

    # Calculate the azimuth
    azim = pi + atan2(sin(ha), cos(ha)*sin(lat) - tan(dec)*cos(lat))

    # Calculate the sine of elevation
    sin_elev = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(ha)

    # Wrap the sine of elevation in the [-1, +1] range
    sin_elev = (sin_elev + 1)%2 - 1

    elev = asin(sin_elev)

    return (azim, elev)


@cython.cdivision(True)
cpdef tuple cyaltAz2RADec(double azim, double elev, double jd, double lat, double lon):
    """ Convert azimuth and altitude in a given time and position on Earth to right ascension and 
        declination. 

    Arguments:
        azim: [float] Azimuth (+east of due north) in radians.
        elev: [float] Elevation above horizon in radians.
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        lon: [float] Longitde of the observer in radians.

    Return:
        (RA, dec): [tuple]
            RA: [float] Right ascension (radians).
            dec: [float] Declination (radians).
    """


    cdef double lst, ha, ra, dec

    # Calculate Local Sidereal Time
    lst = radians(cyjd2LST(jd, degrees(lon)))
    
    # Calculate hour angle
    ha = atan2(-sin(azim), tan(elev)*cos(lat) - cos(azim)*sin(lat))
    
    # Calculate right ascension
    ra = (lst - ha)%(2*pi)

    # Calculate declination
    dec = asin(sin(lat)*sin(elev) + cos(lat)*cos(elev)*cos(azim))

    return (ra, dec)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyraDecToXY(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data, double jd, double lat, double lon, double x_res, \
    double y_res, double az_centre, double alt_centre, double pos_angle_ref, double pix_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_rev, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_rev, str dist_type):
    """ Convert RA, Dec to distorion corrected image coordinates. 

    Arguments:
        RA_data: [ndarray] Array of right ascensions (degrees).
        dec_data: [ndarray] Array of declinations (degrees).
        jd: [float] Julian date.
        lat: [float] Latitude of station in degrees.
        lon: [float] Longitude of station in degrees.
        x_res: [int] X resolution of the camera.
        y_res: [int] Y resolution of the camera.
        az_centre: [float] Azimuth of the FOV centre (degrees).
        alt_centre: [float] Altitude of the FOV centre (degrees).
        pos_angle_ref: [float] Rotation from the celestial meridial (degrees).
        pix_scale: [float] Image scale (px/deg).
        x_poly_rev: [ndarray float] Distortion polynomial in X direction for reverse mapping.
        y_poly_rev: [ndarray float] Distortion polynomail in Y direction for reverse mapping.
        dist_type: [str] Distortion type. Can be: poly3+radial, radial3, or radial5.
    
    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.
    """

    cdef int i
    cdef double ra_star, dec_star, ra_centre, dec_centre
    cdef double ra, dec, ad, radius, sin_ang, cos_ang, theta, x, y, r, dx, dy, dradius

    # Init output arrays
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array = np.zeros_like(ra_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] y_array = np.zeros_like(ra_data)

    # Precalculate some parameters
    cdef double sl = sin(radians(lat))
    cdef double cl = cos(radians(lat))


    # Convert FOV centre to RA/Dec
    ra_centre, dec_centre = cyaltAz2RADec(radians(az_centre), radians(alt_centre), jd, radians(lat), \
        radians(lon))


    # Convert all equatorial coordinates to image coordinates
    for i in range(ra_data.shape[0]):

        ra_star = ra_data[i]
        dec_star = dec_data[i]

        ### Gnomonization of star coordinates to image coordinates ###
        
        ra = radians(ra_star)
        dec = radians(dec_star)

        # Compute the distance from the FOV centre to the sky coordinate
        radius = angularSeparation(ra_star, dec_star, degrees(ra_centre), degrees(dec_centre))
        ad = radians(radius)

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the image vertical
        sin_ang = cos(dec)*sin(ra - ra_centre)/sin(ad)
        cos_ang = (sin(dec) - sin(dec_centre)*cos(ad))/(cos(dec_centre)*sin(ad))
        theta = -degrees(atan2(sin_ang, cos_ang)) + pos_angle_ref - 90.0

        # Calculate the uncorrected image coordinates
        x = radius*cos(radians(theta))*pix_scale
        y = radius*sin(radians(theta))*pix_scale
        r = sqrt(x**2 + y**2)

        ### ###


        # Apply 3rd order polynomial + one radial term distortion
        if dist_type == "poly3+radial":

            # Calculate the distortion in X direction
            dx = (x_poly_rev[0]
                + x_poly_rev[1]*x
                + x_poly_rev[2]*y
                + x_poly_rev[3]*x**2
                + x_poly_rev[4]*x*y
                + x_poly_rev[5]*y**2
                + x_poly_rev[6]*x**3
                + x_poly_rev[7]*x**2*y
                + x_poly_rev[8]*x*y**2
                + x_poly_rev[9]*y**3
                + x_poly_rev[10]*x*r
                + x_poly_rev[11]*y*r)

            # Calculate the distortion in Y direction
            dy = (y_poly_rev[0]
                + y_poly_rev[1]*x
                + y_poly_rev[2]*y
                + y_poly_rev[3]*x**2
                + y_poly_rev[4]*x*y
                + y_poly_rev[5]*y**2
                + y_poly_rev[6]*x**3
                + y_poly_rev[7]*x**2*y
                + y_poly_rev[8]*x*y**2
                + y_poly_rev[9]*y**3
                + y_poly_rev[10]*y*r
                + y_poly_rev[11]*x*r)


        # Apply the 3rd order radial distortion
        elif dist_type == "radial3":

            # Compute the radial shift
            dradius = r*(x_poly_rev[2] + x_poly_rev[3]*r + x_poly_rev[4]*r**2)

            # Use the X array for storing the distortion parameters (index 0 for X offset, 1 for Y offset)
            dx = x_poly_rev[0] + x*dradius
            dy = x_poly_rev[1] + y*dradius

        # Apply the 5th order radial distortion
        elif dist_type == "radial5":

            # Compute the radial shift
            dradius = r*(x_poly_rev[2] + x_poly_rev[3]*r + x_poly_rev[4]*r**2 + x_poly_rev[5]*r**3 \
                + x_poly_rev[6]*r**4)

            # Use the X array for storing the distortion parameters (index 0 for X offset, 1 for Y offset)
            dx = x_poly_rev[0] + x*dradius
            dy = x_poly_rev[1] + y*dradius

        # Apply the 5th order radial distortion
        elif dist_type == "radial7":

            # Compute the radial shift
            dradius = r*(x_poly_rev[2] + x_poly_rev[3]*r + x_poly_rev[4]*r**2 + x_poly_rev[5]*r**3 \
                + x_poly_rev[6]*r**4 + x_poly_rev[7]*r**5 + x_poly_rev[8]*r**6)

            # Use the X array for storing the distortion parameters (index 0 for X offset, 1 for Y offset)
            dx = x_poly_rev[0] + x*dradius
            dy = x_poly_rev[1] + y*dradius

        else:
            dx = 0
            dy = 0


        # Add the distortion correction and calculate X image coordinates
        x_array[i] = x - dx + x_res/2.0

        # Add the distortion correction and calculate Y image coordinates
        y_array[i] = y - dy + y_res/2.0


    return x_array, y_array



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyXYToRADec(np.ndarray[FLOAT_TYPE_t, ndim=1] jd_data, np.ndarray[FLOAT_TYPE_t, ndim=1] x_data, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_data, double lat, double lon, double h0, double x_res, double y_res, \
    double ra_ref, double dec_ref, double pos_angle_ref, double pix_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, str dist_type):
    """
    Arguments:
        jd_data: [ndarray] Julian date of each data point.
        x_data: [ndarray] 1D numpy array containing the image column.
        y_data: [ndarray] 1D numpy array containing the image row.
        lat: [float] Latitude of the observer in degrees.
        lon: [float] Longitde of the observer in degress.
        h0: [float] Reference hour angle (deg).
        x_res: [int] Image size, X dimension (px).
        y_res: [int] Image size, Y dimenstion (px).
        ra_ref: [float] Reference right ascension of the image centre (degrees).
        dec_ref: [float] Reference declination of the image centre (degrees).
        pos_angle_ref: [float] Field rotation parameter (degrees).
        pix_scale: [float] Plate scale (px/deg).
        x_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward X axis polynomial parameters.
        y_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward Y axis polynomial parameters.
        dist_type: [str] Distortion type. Can be: poly3+radial, radial3, or radial5.
    
    Return:
        (ra_data, dec_data): [tuple of ndarrays]
            
            ra_data: [ndarray] Right ascension of each point (deg).
            dec_data: [ndarray] Declination of each point (deg).
            magnitude_data: [ndarray] Array of meteor's lightcurve apparent magnitudes.
    """

    cdef int i
    cdef double jd, x_img, y_img, r_img, dx, x_corr, dy, y_corr, dradius
    cdef double dec_rad, sin_lat, cos_lat
    cdef double radius, theta, sin_t, dec0, cos_t, ra0, h, x, y, z, r, azimuth, altitude
    cdef double ha, ra, dec

    # Convert the reference declination to radians
    dec_rad = radians(dec_ref)

    # Precalculate some parameters
    sin_lat = sin(radians(lat))
    cos_lat = cos(radians(lat))

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data = np.zeros_like(jd_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data = np.zeros_like(jd_data)

    # Go through all given data points and convert them from X, Y to RA, Dec
    for i in range(jd_data.shape[0]):

        # Choose time and image coordiantes
        jd = jd_data[i]
        x_img = x_data[i]
        y_img = y_data[i]


        ### APPLY DISTORTION CORRECTION ###

        # Normalize image coordinates to the image centre and compute the radius from image centre
        x_img = x_img - x_res/2.0
        y_img = y_img - y_res/2.0
        r_img = sqrt(x_img**2 + y_img**2)


        # Apply 3rd order polynomial + one radial term distortion
        if dist_type == "poly3+radial":

            # Compute offset in X direction
            dx = (x_poly_fwd[0]
                + x_poly_fwd[1]*x_img
                + x_poly_fwd[2]*y_img
                + x_poly_fwd[3]*x_img**2
                + x_poly_fwd[4]*x_img*y_img
                + x_poly_fwd[5]*y_img**2
                + x_poly_fwd[6]*x_img**3
                + x_poly_fwd[7]*x_img**2*y_img
                + x_poly_fwd[8]*x_img*y_img**2
                + x_poly_fwd[9]*y_img**3
                + x_poly_fwd[10]*x_img*r_img
                + x_poly_fwd[11]*y_img*r_img)

            # Compute offset in Y direction
            dy = (y_poly_fwd[0]
                + y_poly_fwd[1]*x_img
                + y_poly_fwd[2]*y_img
                + y_poly_fwd[3]*x_img**2
                + y_poly_fwd[4]*x_img*y_img
                + y_poly_fwd[5]*y_img**2
                + y_poly_fwd[6]*x_img**3
                + y_poly_fwd[7]*x_img**2*y_img
                + y_poly_fwd[8]*x_img*y_img**2
                + y_poly_fwd[9]*y_img**3
                + y_poly_fwd[10]*y_img*r_img
                + y_poly_fwd[11]*x_img*r_img)


        # Apply the 3rd order radial distortion
        elif dist_type == "radial3":

            # Compute the radial shift
            dradius = r_img*(x_poly_fwd[2] + x_poly_fwd[3]*r_img + x_poly_fwd[4]*r_img**2)

            # Use the X array for storing the distortion parameters (index 0 for X offset, 1 for Y offset)
            dx = x_poly_fwd[0] + x_img*dradius
            dy = x_poly_fwd[1] + y_img*dradius


        # Apply the 5th order radial distortion
        elif dist_type == "radial5":

            # Compute the radial shift
            dradius = r_img*(x_poly_fwd[2] + x_poly_fwd[3]*r_img + x_poly_fwd[4]*r_img**2 \
                + x_poly_fwd[5]*r_img**3 + x_poly_fwd[6]*r_img**4)

            # Use the X array for storing the distortion parameters (index 0 for X offset, 1 for Y offset)
            dx = x_poly_fwd[0] + x_img*dradius
            dy = x_poly_fwd[1] + y_img*dradius

        # Apply the 5th order radial distortion
        elif dist_type == "radial7":

            # Compute the radial shift
            dradius = r_img*(x_poly_fwd[2] + x_poly_fwd[3]*r_img + x_poly_fwd[4]*r_img**2 \
                + x_poly_fwd[5]*r_img**3 + x_poly_fwd[6]*r_img**4 + x_poly_fwd[7]*r_img**5 \
                + x_poly_fwd[8]*r_img**6)

            # Use the X array for storing the distortion parameters (index 0 for X offset, 1 for Y offset)
            dx = x_poly_fwd[0] + x_img*dradius
            dy = x_poly_fwd[1] + y_img*dradius

        else:
            dx = 0
            dy = 0


        # Correct image coordinates for distortion
        y_corr = y_img + dy
        x_corr = x_img + dx

        # Gnomonize coordinates
        x_corr = x_corr/pix_scale
        y_corr = y_corr/pix_scale

        ### ###


        ### Convert gnomonic X, Y to alt, az ###

        # Radius from FOV centre to sky coordinate
        radius = radians(sqrt(x_corr**2 + y_corr**2))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the image vertical
        theta = (pi/2 - radians(pos_angle_ref) + atan2(y_corr, x_corr))%(2*pi)

        # Transform the radius and direction to coordinates on the sky
        sin_t = sin(dec_rad)*cos(radius) + cos(dec_rad)*sin(radius)*cos(theta)
        dec0 = atan2(sin_t, sqrt(1 - sin_t**2))

        sin_t = sin(theta)*sin(radius)/cos(dec0)
        cos_t = (cos(radius) - sin(dec0)*sin(dec_rad))/(cos(dec0)*cos(dec_rad))
        ra0 = (ra_ref - degrees(atan2(sin_t, cos_t)))%360

        # Add the hour angle difference to the right ascension
        ra = (ra0 + cyjd2LST(jd, 0) - h0)%360

        # Convert declination to degrees
        dec = degrees(dec0)


        # Assign values to output list
        ra_data[i] = ra
        dec_data[i] = dec


    return ra_data, dec_data