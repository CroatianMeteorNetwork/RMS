#!/usr/bin/env python3
"""
Test XY → RA/Dec → Alt/Az → XY roundtrip at various points.
"""

import sys
import os
import numpy as np

# Add RMS to path
sys.path.insert(0, '/home/luc/source/RMS')

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, raDecToXYPP_iter, AltAzToXYPP, xyToAltAzPP
from RMS.Astrometry.Conversions import jd2Date
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz

def test_xy_roundtrip():
    """Test roundtrip conversions at various image points."""
    
    # Load platepar
    pp = Platepar()
    pp.read('/home/luc/data/advect_staging/US0001/US0001_20250707_031251_to_20250707_111038/platepar_cmn2010.cal')
    
    print("XY ROUNDTRIP TEST")
    print("="*70)
    print(f"Image size: {pp.X_res} x {pp.Y_res}")
    print(f"Distortion center: ({pp.x_poly_fwd[0]:.2f}, {pp.y_poly_fwd[0]:.2f})")
    print(f"JD: {pp.JD:.6f}")
    
    # Get time as tuple for xyToRaDecPP
    time_data = [jd2Date(pp.JD)]
    
    # Test points
    test_points = [
        ("Image center", pp.X_res/2, pp.Y_res/2),
        ("Top-left corner", 50, 50),
        ("Top-right corner", pp.X_res - 50, 50),
        ("Bottom-left corner", 50, pp.Y_res - 50),
        ("Bottom-right corner", pp.X_res - 50, pp.Y_res - 50),
        ("Mid-left edge", 50, pp.Y_res/2),
        ("Mid-right edge", pp.X_res - 50, pp.Y_res/2),
    ]
    
    print("\n" + "="*70)
    print("TEST 1: XY → RA/Dec → XY")
    print("-"*70)
    
    for name, x, y in test_points:
        print(f"\n{name}: ({x:.1f}, {y:.1f})")
        
        # XY to RA/Dec
        _, ra_list, dec_list, _ = xyToRaDecPP(
            time_data, [x], [y], [1], pp, extinction_correction=False
        )
        ra = ra_list[0]
        dec = dec_list[0]
        print(f"  → RA/Dec: {ra:.6f}°, {dec:.6f}°")
        
        # RA/Dec back to XY
        x_arr, y_arr = raDecToXYPP_iter(np.array([ra]), np.array([dec]), pp.JD, pp)
        x_back = x_arr[0]
        y_back = y_arr[0]
        print(f"  → XY: ({x_back:.3f}, {y_back:.3f})")
        
        error = np.sqrt((x_back - x)**2 + (y_back - y)**2)
        print(f"  Error: {error:.3f} px")
    
    print("\n" + "="*70)
    print("TEST 2: XY → Alt/Az → XY (direct)")
    print("-"*70)
    
    for name, x, y in test_points:
        print(f"\n{name}: ({x:.1f}, {y:.1f})")
        
        # XY to Alt/Az (direct)
        alt_list, az_list = xyToAltAzPP([x], [y], pp)  # Returns (Alt, Az)
        alt = alt_list[0]
        az = az_list[0]
        print(f"  → Alt/Az: {alt:.6f}°, {az:.6f}°")
        
        # Alt/Az back to XY (direct)
        x_arr, y_arr = AltAzToXYPP(np.array([alt]), np.array([az]), pp)  # Expects (alt, az)
        x_back = x_arr[0]
        y_back = y_arr[0]
        print(f"  → XY: ({x_back:.3f}, {y_back:.3f})")
        
        error = np.sqrt((x_back - x)**2 + (y_back - y)**2)
        print(f"  Error: {error:.3f} px")
    
    print("\n" + "="*70)
    print("TEST 3: XY → RA/Dec → Alt/Az → XY (full chain)")
    print("-"*70)
    
    for name, x, y in test_points:
        print(f"\n{name}: ({x:.1f}, {y:.1f})")
        
        # XY to RA/Dec
        _, ra_list, dec_list, _ = xyToRaDecPP(
            time_data, [x], [y], [1], pp, extinction_correction=False
        )
        ra = ra_list[0]
        dec = dec_list[0]
        print(f"  → RA/Dec: {ra:.6f}°, {dec:.6f}°")
        
        # RA/Dec to Alt/Az
        # Use the Cython function directly with proper radian conversion
        az, alt = cyTrueRaDec2ApparentAltAz(np.radians(ra), np.radians(dec), 
                                            pp.JD, np.radians(pp.lat), np.radians(pp.lon), 
                                            False)
        # Convert output from radians to degrees
        az = np.degrees(az)
        alt = np.degrees(alt)
        print(f"  → Alt/Az: {alt:.6f}°, {az:.6f}°")
        
        # Alt/Az to XY (direct)
        x_arr, y_arr = AltAzToXYPP(np.array([alt]), np.array([az]), pp)  # Expects (alt, az)
        x_back = x_arr[0]
        y_back = y_arr[0]
        print(f"  → XY: ({x_back:.3f}, {y_back:.3f})")
        
        error = np.sqrt((x_back - x)**2 + (y_back - y)**2)
        print(f"  Error: {error:.3f} px")
        
        if error > 1.0:
            print(f"  ⚠️ Large error!")
    
    print("\n" + "="*70)
    print("TEST 4: Compare two-step vs direct Alt/Az")
    print("-"*70)
    
    for name, x, y in test_points:
        print(f"\n{name}: ({x:.1f}, {y:.1f})")
        
        # Method 1: XY → Alt/Az (direct)
        alt_direct_list, az_direct_list = xyToAltAzPP([x], [y], pp)  # Returns (Alt, Az)
        alt_direct = alt_direct_list[0]
        az_direct = az_direct_list[0]
        
        # Method 2: XY → RA/Dec → Alt/Az
        _, ra_list, dec_list, _ = xyToRaDecPP(
            [jd2Date(pp.JD)], [x], [y], [1], pp, extinction_correction=False
        )
        ra = ra_list[0]
        dec = dec_list[0]
        # Use the Cython function directly with proper radian conversion
        az_twostep, alt_twostep = cyTrueRaDec2ApparentAltAz(np.radians(ra), np.radians(dec), 
                                                            pp.JD, np.radians(pp.lat), np.radians(pp.lon), 
                                                            pp.refraction)
        # Convert output from radians to degrees
        az_twostep = np.degrees(az_twostep)
        alt_twostep = np.degrees(alt_twostep)
        
        print(f"  Direct:   Alt={alt_direct:.6f}°, Az={az_direct:.6f}°")
        print(f"  Two-step: Alt={alt_twostep:.6f}°, Az={az_twostep:.6f}°")
        print(f"  Difference: ΔAlt={alt_twostep - alt_direct:.6f}°, ΔAz={az_twostep - az_direct:.6f}°")
        
        if np.sqrt(abs(az_twostep - az_direct)**2 +  abs(alt_twostep - alt_direct)**2) > 2/60:
            print(f"  ⚠️ Methods disagree!")

if __name__ == "__main__":
    test_xy_roundtrip()