#!/usr/bin/env python3
"""
Test the iterative rotation calculation without infinite recursion.
"""

import sys
import os
import numpy as np
import copy
import scipy.optimize

# Add RMS to path
sys.path.insert(0, '/home/luc/source/RMS')

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, rotationWrtHorizon
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz, cyAltAzToXY, cyraDec2AltAz, refractionTrueToApparent

def rotationWrtHorizon_iter_safe(platepar):
    """ Safe version of rotationWrtHorizon_iter that avoids circular dependency.
    
    Arguments:
        platepar: [Platepar object] Input platepar.
        
    Return:
        opt_rot_angle: [float] Optimized rotation w.r.t. horizon (degrees).
    """

    # Deep copy of platepar
    platepar_temp = copy.deepcopy(platepar)
    
    # Create a 10x10 grid on the image, avoiding the edges
    margin = 30
    x_grid, y_grid = np.linspace(margin, platepar_temp.X_res - margin, 10), np.linspace(margin, platepar_temp.Y_res - margin, 10)
    xx, yy = np.meshgrid(x_grid, y_grid)
    xx, yy = xx.flatten(), yy.flatten()
    
    # Compute celestial coordinates for each grid point
    from RMS.Astrometry.Conversions import jd2Date
    time_data = [jd2Date(platepar_temp.JD)]
    _, ra_arr, dec_arr, _ = xyToRaDecPP(time_data, xx, yy, len(xx)*[1], platepar_temp, extinction_correction=False)
    
    # Compute the alt/az center once (outside the optimization)
    az_centre, alt_centre = cyraDec2AltAz(
        np.radians(platepar_temp.RA_d),
        np.radians(platepar_temp.dec_d),
        platepar_temp.JD,
        np.radians(platepar_temp.lat),
        np.radians(platepar_temp.lon)
    )
    alt_centre = refractionTrueToApparent(alt_centre)
    az_centre, alt_centre = np.degrees(az_centre), np.degrees(alt_centre)
    
    # Objective function for optimization
    def objective(rot_angle):
        total_error = 0
        for i, (ra, dec) in enumerate(zip(ra_arr, dec_arr)):
            # Convert RA/Dec to Alt/Az
            az, alt = cyraDec2AltAz(np.radians(ra), np.radians(dec), 
                                   platepar_temp.JD, np.radians(platepar_temp.lat), 
                                   np.radians(platepar_temp.lon))
            az = np.degrees(az)
            alt = np.degrees(alt)
            
            # Convert Alt/Az to XY using the Cython function directly
            x_out, y_out = cyAltAzToXY(np.array([alt]), np.array([az]),
                float(platepar_temp.X_res), float(platepar_temp.Y_res), float(az_centre),
                float(alt_centre), float(rot_angle[0]), platepar_temp.F_scale, platepar_temp.x_poly_fwd,
                platepar_temp.y_poly_fwd, str(platepar_temp.distortion_type), refraction=platepar_temp.refraction,
                equal_aspect=platepar_temp.equal_aspect, force_distortion_centre=platepar_temp.force_distortion_centre,
                asymmetry_corr=platepar_temp.asymmetry_corr)
            
            # Calculate error
            error = np.sqrt((x_out[0] - xx[i])**2 + (y_out[0] - yy[i])**2)
            total_error += error
        
        return total_error
    
    # Initial guess for rotation_from_horiz
    initial_guess = [0]
    
    # Optimize
    result = scipy.optimize.minimize(objective, initial_guess, method='Nelder-Mead')
    
    opt_rot_angle = result.x[0]
    
    return opt_rot_angle

def test_rotation():
    """Test the rotation calculation."""
    
    # Load platepar
    pp = Platepar()
    pp.read('/home/luc/data/advect_staging/US0001/US0001_20250707_031251_to_20250707_111038/platepar_cmn2010.cal')
    
    print("ROTATION CALCULATION TEST")
    print("="*70)
    
    # Test the non-iterative version
    rot_standard = rotationWrtHorizon(pp)
    print(f"Standard rotation: {rot_standard:.6f}°")
    
    # Test the safe iterative version
    rot_iter = rotationWrtHorizon_iter_safe(pp)
    print(f"Iterative rotation: {rot_iter:.6f}°")
    
    print(f"Difference: {rot_iter - rot_standard:.6f}°")

if __name__ == "__main__":
    test_rotation()