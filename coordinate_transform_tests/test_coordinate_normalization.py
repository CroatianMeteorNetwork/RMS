#!/usr/bin/env python3
"""
Compare coordinate normalization differences between cyAltAzToXY and cyRaDecToXY_iter.

KEY DIFFERENCES FOUND:

1. Distortion Center Handling:
   - cyRaDecToXY_iter: When force_distortion_centre=True, adds 0.5 pixel offset:
     x0 = 0.5/(x_res/2.0)  # Line 1433
     y0 = 0.5/(y_res/2.0)  # Line 1434
   
   - cyAltAzToXY: When force_distortion_centre=True, sets to 0:
     x0 = 0.0  # Line 2110
     y0 = 0.0  # Line 2111

2. Normalization Order:
   - cyRaDecToXY_iter: 
     * Normalizes offsets AFTER reading them (lines 1442-1443)
     * x0 *= (x_res/2.0)
     * y0 *= (y_res/2.0)
   
   - cyAltAzToXY:
     * Normalizes offsets AFTER setting all parameters (lines 2158-2159)
     * Same normalization: x0 *= (x_res/2.0), y0 *= (y_res/2.0)

3. Final Coordinate Transformation:
   - cyRaDecToXY_iter (lines 1654-1655):
     * x_array[i] = x_img + x_res/2.0
     * y_array[i] = y_img + y_res/2.0
   
   - cyAltAzToXY (lines 2376-2377):
     * x_img = x_img + x0 + x_res/2.0
     * y_img = y_img + y0 + y_res/2.0
     * NOTE: Adds x0, y0 offsets here!

4. Iterative Solver Differences:
   - cyRaDecToXY_iter: Uses different iteration approach with:
     * dx = (x_img - x0)*r_scale - x0
     * dy = (y_img - y0)*r_scale*(1.0 + xy) - y0*(1.0 + xy) + y_img*xy
   
   - cyAltAzToXY: Uses simpler radial iteration

The main issue appears to be in step 3: cyAltAzToXY adds x0, y0 in the final transformation,
while cyRaDecToXY_iter doesn't. This could cause coordinate shifts.
"""

import sys
import os
import numpy as np

# Add RMS to path
sys.path.insert(0, '/home/luc/source/RMS')

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP_iter, AltAzToXYPP, xyToAltAzPP
from RMS.Astrometry.Conversions import jd2Date
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz

def test_normalization():
    """Test coordinate normalization differences."""
    
    # Load platepar
    pp = Platepar()
    pp.read('/home/luc/data/advect_staging/US0001/US0001_20250707_031251_to_20250707_111038/platepar_cmn2010.cal')
    
    print("COORDINATE NORMALIZATION TEST")
    print("="*70)
    print(f"Image size: {pp.X_res} x {pp.Y_res}")
    print(f"Distortion center from platepar: ({pp.x_poly_fwd[0]:.6f}, {pp.y_poly_fwd[0]:.6f})")
    print(f"  In pixels: ({pp.x_poly_fwd[0] * pp.X_res/2:.2f}, {pp.y_poly_fwd[0] * pp.Y_res/2:.2f})")
    print(f"Force distortion center: {pp.force_distortion_centre}")
    
    # Get time as tuple for xyToRaDecPP
    time_data = [jd2Date(pp.JD)]
    
    # Test at image center
    x, y = pp.X_res/2, pp.Y_res/2
    
    print(f"\nTest point: ({x:.1f}, {y:.1f})")
    
    # Path 1: XY → RA/Dec → XY (using cyRaDecToXY_iter)
    print("\n" + "-"*70)
    print("PATH 1: XY → RA/Dec → XY (cyRaDecToXY_iter)")
    
    _, ra_list, dec_list, _ = xyToRaDecPP(
        time_data, [x], [y], [1], pp, extinction_correction=False
    )
    ra = ra_list[0]
    dec = dec_list[0]
    print(f"  RA/Dec: {ra:.6f}°, {dec:.6f}°")
    
    x_arr, y_arr = raDecToXYPP_iter(np.array([ra]), np.array([dec]), pp.JD, pp)
    x_back = x_arr[0]
    y_back = y_arr[0]
    print(f"  XY back: ({x_back:.3f}, {y_back:.3f})")
    print(f"  Error: {np.sqrt((x_back - x)**2 + (y_back - y)**2):.3f} px")
    
    # Path 2: XY → Alt/Az → XY (using cyAltAzToXY)
    print("\n" + "-"*70)
    print("PATH 2: XY → Alt/Az → XY (cyAltAzToXY)")
    
    alt_list, az_list = xyToAltAzPP([x], [y], pp)
    alt = alt_list[0]
    az = az_list[0]
    print(f"  Alt/Az: {alt:.6f}°, {az:.6f}°")
    
    x_arr, y_arr = AltAzToXYPP(np.array([alt]), np.array([az]), pp)
    x_back = x_arr[0]
    y_back = y_arr[0]
    print(f"  XY back: ({x_back:.3f}, {y_back:.3f})")
    print(f"  Error: {np.sqrt((x_back - x)**2 + (y_back - y)**2):.3f} px")
    
    # Path 3: Mixed path - RA/Dec → Alt/Az → XY
    print("\n" + "-"*70)
    print("PATH 3: RA/Dec → Alt/Az → XY (mixed)")
    
    # Use the RA/Dec from path 1
    az_mixed, alt_mixed = cyTrueRaDec2ApparentAltAz(
        np.radians(ra), np.radians(dec), 
        pp.JD, np.radians(pp.lat), np.radians(pp.lon), 
        False  # No refraction for consistency
    )
    az_mixed = np.degrees(az_mixed)
    alt_mixed = np.degrees(alt_mixed)
    print(f"  Alt/Az from RA/Dec: {alt_mixed:.6f}°, {az_mixed:.6f}°")
    
    x_arr, y_arr = AltAzToXYPP(np.array([alt_mixed]), np.array([az_mixed]), pp)
    x_mixed = x_arr[0]
    y_mixed = y_arr[0]
    print(f"  XY from Alt/Az: ({x_mixed:.3f}, {y_mixed:.3f})")
    print(f"  Error vs original: {np.sqrt((x_mixed - x)**2 + (y_mixed - y)**2):.3f} px")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("-"*70)
    print("The issue is likely in cyAltAzToXY adding x0, y0 offsets in final transformation")
    print("while cyRaDecToXY_iter doesn't. This causes systematic shifts in Alt/Az → XY.")

if __name__ == "__main__":
    test_normalization()