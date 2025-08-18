""" CAMS type CAL file, used for CAMS compatibility, but not operationally for RMS. """

from __future__ import print_function, division, absolute_import


import os
import copy
import math


from RMS.GeoidHeightEGM96 import mslToWGS84Height
from RMS.Astrometry.Conversions import jd2Date
from RMS.Astrometry.ApplyAstrometry import rotationWrtHorizon, rotationWrtStandard



def writeCAL(night_dir, config, platepar):
    """ Write the CAL file. 

    Arguments:
        night_dir: [str] Path of the night directory where the file will be saved. This folder will be used
            to construct the name of CAL file.
        config: [Config]
        platepar: [Platepar]

    Return:
        file_name: [str] Name of the CAL file.

    """

    # Remove the last slash, if it exists
    if night_dir[-1] == os.sep:
        night_dir = night_dir[:-1]

    # Extract time from night name
    _, night_name = os.path.split(night_dir)
    night_time = "_".join(night_name.split('_')[1:4])[:-3]

    # Construct the CAL file name
    file_name = "CAL_{:06d}_{:s}.txt".format(config.cams_code, night_time)


    # If there was no platepar, don't create a CAL file
    if platepar is None:
        print("No Platepar available to create a CAL file!")
        return None

    # Make a copy of the platepar that can be modified
    platepar = copy.deepcopy(platepar)


    # Compute rotations (must be done before distortion correction)
    rot_horiz = rotationWrtHorizon(platepar)
    rot_std = rotationWrtStandard(platepar)


    if platepar.distortion_type == "poly3+radial":

        # CAMS code for the distortion type
        distortion_code = "201"

        # Switch ry in Y coeffs
        platepar.y_poly_fwd[11], platepar.y_poly_fwd[10] = platepar.y_poly_fwd[10], platepar.y_poly_fwd[11]


        # Correct distortion parameters so they are CAMS compatible
        platepar.x_poly_fwd[ 1] = +platepar.x_poly_fwd[ 1] + 1.0
        platepar.x_poly_fwd[ 2] = -platepar.x_poly_fwd[ 2]
        platepar.x_poly_fwd[ 4] = -platepar.x_poly_fwd[ 4]
        platepar.x_poly_fwd[ 7] = -platepar.x_poly_fwd[ 7]
        platepar.x_poly_fwd[ 9] = -platepar.x_poly_fwd[ 9]
        platepar.x_poly_fwd[11] = -platepar.x_poly_fwd[11]
        platepar.y_poly_fwd[ 2] = -platepar.y_poly_fwd[ 2] - 1.0
        platepar.y_poly_fwd[ 4] = -platepar.y_poly_fwd[ 4]
        platepar.y_poly_fwd[ 7] = -platepar.y_poly_fwd[ 7]
        platepar.y_poly_fwd[ 9] = -platepar.y_poly_fwd[ 9]
        platepar.y_poly_fwd[11] = -platepar.y_poly_fwd[11]

    else:

        if platepar.distortion_type == "radial3-odd":
            distortion_code = "203"
        elif platepar.distortion_type == "radial3-all":
            distortion_code = "204"
        elif platepar.distortion_type == "radial5-odd":
            distortion_code = "205"
        elif platepar.distortion_type == "radial4-all":
            distortion_code = "206"
        elif platepar.distortion_type == "radial7-odd":
            distortion_code = "207"
        elif platepar.distortion_type == "radial5-all":
            distortion_code = "208"
        elif platepar.distortion_type == "radial9-odd":
            distortion_code = "209"
        else:
            distortion_code = "201"
        

        # Reset the distortion to zero out all distortion params
        platepar.setDistortionType("poly3+radial")


    # Compute scale in arcmin/px
    arcminperpixel = 60/platepar.F_scale

    # Correct scaling and rotation
    for k in range(12):
        
        x_prime = platepar.x_poly_fwd[k]*math.radians(arcminperpixel/60.0)
        y_prime = platepar.y_poly_fwd[k]*math.radians(arcminperpixel/60.0)

        platepar.x_poly_fwd[k] = math.cos(math.radians(platepar.pos_angle_ref))*x_prime \
            + math.sin(math.radians(platepar.pos_angle_ref))*y_prime

        platepar.y_poly_fwd[k] = math.sin(math.radians(platepar.pos_angle_ref))*x_prime \
            - math.cos(math.radians(platepar.pos_angle_ref))*y_prime

    # Compute WGS84 height
    height_wgs84 = mslToWGS84Height(math.radians(platepar.lat), math.radians(platepar.lon),
                                    platepar.elev, config)
    print(platepar.lat, platepar.lon, platepar.elev)
    print("Height wgs84 = ", height_wgs84)

    # Open the file
    with open(os.path.join(night_dir, file_name), 'w') as f:

        # Construct calibration date and time
        calib_dt = jd2Date(platepar.JD, dt_obj=True)
        calib_date = calib_dt.strftime("%m/%d/%Y")
        calib_time = calib_dt.strftime("%H:%M:%S.%f")[:-3]

        s  =" Camera number            = {:d}\n".format(config.cams_code)
        s +=" Calibration date         = {:s}\n".format(calib_date)
        s +=" Calibration time (UT)    = {:s}\n".format(calib_time)
        s +=" Longitude +west (deg)    = {:9.5f}\n".format(-platepar.lon)
        s +=" Latitude +north (deg)    = {:9.5f}\n".format(platepar.lat)
        s +=" Height above WGS84 (km)  = {:8.4f}\n".format(height_wgs84/1000)
        s +=" FOV dimension hxw (deg)  =   {:.2f} x   {:.2f}\n".format(platepar.fov_v, platepar.fov_h)
        s +=" Plate scale (arcmin/pix) = {:8.3f}\n".format(arcminperpixel)
        s +=" Plate roll wrt Std (deg) = {:8.3f}\n".format(rot_std)
        s +=" Cam tilt wrt Horiz (deg) = {:8.3f}\n".format(rot_horiz)
        s +=" Frame rate (Hz)          = {:8.3f}\n".format(config.fps)
        s +=" Cal center RA (deg)      = {:8.3f}\n".format(platepar.RA_d)
        s +=" Cal center Dec (deg)     = {:8.3f}\n".format(platepar.dec_d)
        s +=" Cal center Azim (deg)    = {:8.3f}\n".format(platepar.az_centre)
        s +=" Cal center Elev (deg)    = {:8.3f}\n".format(platepar.alt_centre)
        s +=" Cal center col (colcen)  = {:8.3f}\n".format(platepar.X_res/2)
        s +=" Cal center row (rowcen)  = {:8.3f}\n".format(platepar.Y_res/2)
        s +=" Cal fit order            = {:>4s}\n".format(distortion_code)
        s +="\n"
        s +=" Camera description       = None\n"
        s +=" Lens description         = None\n"
        s +=" Focal length (mm)        =    0.000\n"
        s +=" Focal ratio              =    0.000\n"
        s +=" Pixel pitch H (um)       =    0.000\n"
        s +=" Pixel pitch V (um)       =    0.000\n"
        s +=" Spectral response B      = {:8.3f}\n".format(config.star_catalog_band_ratios[0])
        s +=" Spectral response V      = {:8.3f}\n".format(config.star_catalog_band_ratios[1])
        s +=" Spectral response R      = {:8.3f}\n".format(config.star_catalog_band_ratios[2])
        s +=" Spectral response I      = {:8.3f}\n".format(config.star_catalog_band_ratios[3])
        s +=" Vignetting coef(deg/pix) =    0.000\n"
        s +=" Gamma                    = {:8.3f}\n".format(config.gamma)
        s +="\n"
        s +=" Xstd, Ystd = Radialxy2Standard( col, row, colcen, rowcen, Xcoef, Ycoef )\n"
        s +=" x = col - colcen\n"
        s +=" y = rowcen - row\n"
        s +="\n"
        s +=" Term       Xcoef            Ycoef     \n"
        s +=" ----  ---------------  ---------------\n"
        s +=" 1     {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[0], platepar.y_poly_fwd[0])
        s +=" x     {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[1], platepar.y_poly_fwd[1])
        s +=" y     {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[2], platepar.y_poly_fwd[2])
        s +=" xx    {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[3], platepar.y_poly_fwd[3])
        s +=" xy    {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[4], platepar.y_poly_fwd[4])
        s +=" yy    {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[5], platepar.y_poly_fwd[5])
        s +=" xxx   {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[6], platepar.y_poly_fwd[6])
        s +=" xxy   {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[7], platepar.y_poly_fwd[7])
        s +=" xyy   {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[8], platepar.y_poly_fwd[8])
        s +=" yyy   {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[9], platepar.y_poly_fwd[9])
        s +=" rx    {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[10], platepar.y_poly_fwd[10])
        s +=" ry    {:+.7e}    {:+.7e} \n".format(platepar.x_poly_fwd[11], platepar.y_poly_fwd[11])
        s +=" ----  ---------------  ---------------\n"
        s +="\n"
        s +=" Mean O-C =   0.000 +-   0.000 arcmin\n"
        s +="\n"
        s +=" Magnitude = A + B (logI-logVig)   fit mV vs. -2.5 (logI-logVig),   B-V <  1.20, mV <  6.60\n"
        s +="         A = {:8.3f} \n".format(platepar.mag_lev)
        s +="         B =   -2.50 \n"
        s +="\n"
        s +=" Magnitude = -2.5 ( C + D (logI-logVig) )   fit logFlux vs. Gamma (logI-logVig), mV <  6.60\n"
        s +="         C = {:8.3f} \n".format(platepar.mag_lev/(-2.5))
        s +="         D =    1.00 \n"
        s +="\n"
        s +=" logVig = log( cos( Vignetting_coef * Rpixels * pi/180 )^4 )\n"
        s +="\n"
        s +="\n"
        s +=" Star    RA (deg)  DEC (deg)    row      col       V      B-V      R      IR    logInt  logVig  logFlux  O-C arcmin \n"
        s +=" ----   ---------  ---------  -------  -------  ------  ------  ------  ------  ------  ------  -------  ---------- \n"


        # Write CAL content
        f.write(s)


        return file_name



if __name__ == "__main__":


    import argparse
    import RMS.ConfigReader as cr
    from RMS.Formats.Platepar import Platepar


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Covert the RMS platepar to a CAMS-style CAL file.")

    arg_parser.add_argument('platepar_path', metavar='PLATEPAR_PATH', type=str, \
        help='Path to a platepar file.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Extract parent directory
    dir_path = os.path.dirname(cml_args.platepar_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Load a platepar file
    pp = Platepar()
    pp.read(cml_args.platepar_path, use_flat=config.use_flat)

    # Write the CAL file
    writeCAL(dir_path, config, pp)
