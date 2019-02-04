""" CAMS type CAL file, used for CAMS compatibility, but not operationally for RMS. """

from __future__ import print_function, division, absolute_import


import os
from RMS.Astrometry.Conversions import jd2Date
from RMS.Formats.Platepar import Platepar

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


    # If there was no platepar, init an empty one
    if platepar is None:
        platepar = Platepar()

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
        s +=" Height above WGS84 (km)  = {:8.5f}\n".format(platepar.elev/1000)
        s +=" FOV dimension hxw (deg)  =   {:.2f} x   {:.2f}\n".format(platepar.fov_h, platepar.fov_v)
        s +=" Plate scale (arcmin/pix) = {:8.3f}\n".format(60/platepar.F_scale)
        s +=" Plate roll wrt Std (deg) = {:8.3f}\n".format(platepar.pos_angle_ref)
        s +=" Cam tilt wrt Horiz (deg) =    0.000\n"
        s +=" Frame rate (Hz)          = {:8.3f}\n".format(config.fps)
        s +=" Cal center RA (deg)      = {:8.3f}\n".format(platepar.RA_d)
        s +=" Cal center Dec (deg)     = {:8.3f}\n".format(platepar.dec_d)
        s +=" Cal center Azim (deg)    = {:8.3f}\n".format(platepar.az_centre)
        s +=" Cal center Elev (deg)    = {:8.3f}\n".format(platepar.alt_centre)
        s +=" Cal center col (colcen)  = {:8.3f}\n".format(platepar.X_res/2)
        s +=" Cal center row (rowcen)  = {:8.3f}\n".format(platepar.Y_res/2)
        s +=" Cal fit order            = 201\n" # 201 = RMS 3rd order poly with radial terms
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
        s +=" 1     {:+.8e}  {:+.8e} \n".format(platepar.x_poly[0], platepar.y_poly[0])
        s +=" x     {:+.8e}  {:+.8e} \n".format(platepar.x_poly[1], platepar.y_poly[1])
        s +=" y     {:+.8e}  {:+.8e} \n".format(platepar.x_poly[2], platepar.y_poly[2])
        s +=" xx    {:+.8e}  {:+.8e} \n".format(platepar.x_poly[3], platepar.y_poly[3])
        s +=" xy    {:+.8e}  {:+.8e} \n".format(platepar.x_poly[4], platepar.y_poly[4])
        s +=" yy    {:+.8e}  {:+.8e} \n".format(platepar.x_poly[5], platepar.y_poly[5])
        s +=" xxx   {:+.8e}  {:+.8e} \n".format(platepar.x_poly[6], platepar.y_poly[6])
        s +=" xxy   {:+.8e}  {:+.8e} \n".format(platepar.x_poly[7], platepar.y_poly[7])
        s +=" xyy   {:+.8e}  {:+.8e} \n".format(platepar.x_poly[8], platepar.y_poly[8])
        s +=" yyy   {:+.8e}  {:+.8e} \n".format(platepar.x_poly[9], platepar.y_poly[9])
        s +=" rx    {:+.8e}  {:+.8e} \n".format(platepar.x_poly[10], platepar.y_poly[11])
        s +=" ry    {:+.8e}  {:+.8e} \n".format(platepar.x_poly[11], platepar.y_poly[10])
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

    import RMS.ConfigReader as cr

    # Load the default configuration file
    config = cr.parse(".config")

    # Load a platepar file
    pp = Platepar()
    #pp.read("/mnt/bulk/2018Draconids/ConfirmedFiles/HR0010_20181008_170022_956418_detected/platepar_cmn2010.cal")


    #night_dir = "/mnt/bulk/2018Draconids/ConfirmedFiles/HR0010_20181008_170022_956418_detected"
    night_dir = "D:/Dropbox/RPi_Meteor_Station/data/CA0004_20180516_040459_588816_detected"

    # Write the CAL file
    writeCAL(night_dir, config, pp)