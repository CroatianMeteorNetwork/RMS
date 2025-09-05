#!/usr/bin/env python3
"""
Test the ENU (East-North-Up) coordinate conversion functions.
"""

import sys
import os
import numpy as np

# Add RMS to path
sys.path.insert(0, '/home/luc/source/RMS')

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyHtToENUPP, geoToXYPP_iter, xyToGeoPP_iter
from RMS.Astrometry.Conversions import geo2Cartesian, cartesian2Geo

def test_enu_functions():
    """Test ENU coordinate conversions."""
    
    # Load platepar
    pp = Platepar()
    pp.read('/home/luc/data/advect_staging/US0001/US0001_20250707_031251_to_20250707_111038/platepar_cmn2010.cal')
    
    print("ENU COORDINATE CONVERSION TEST")
    print("="*70)
    print(f"Station: Lat={pp.lat:.6f}°, Lon={pp.lon:.6f}°, Elev={pp.elev:.1f}m")
    print(f"Image size: {pp.X_res} x {pp.Y_res}")
    print(f"JD: {pp.JD:.6f}")
    
    # Test points
    test_points = [
        ("Image center", pp.X_res/2, pp.Y_res/2),
        ("Top-left corner", 50, 50),
        ("Top-right corner", pp.X_res - 50, 50),
        ("Bottom-left corner", 50, pp.Y_res - 50),
        ("Bottom-right corner", pp.X_res - 50, pp.Y_res - 50),
    ]
    
    # Test at different heights
    test_heights = [100000, 50000, 25000]  # meters
    
    print("\n" + "="*70)
    print("TEST 1: XY → ENU at various heights")
    print("-"*70)
    
    for height in test_heights:
        print(f"\nHeight: {height/1000:.0f} km")
        for name, x, y in test_points[:2]:  # Just test first two points for brevity
            print(f"\n  {name}: ({x:.1f}, {y:.1f})")
            
            # XY to ENU
            E, N, U, Eu, Nu, Uu, az, el = xyHtToENUPP(
                np.array([x]), np.array([y]), pp.JD, height, pp
            )
            
            print(f"    ENU position: E={E[0]/1000:.2f}km, N={N[0]/1000:.2f}km, U={U[0]/1000:.2f}km")
            print(f"    Ray direction: Eu={Eu[0]:.4f}, Nu={Nu[0]:.4f}, Uu={Uu[0]:.4f}")
            print(f"    Ray az/el: {np.degrees(az[0]):.2f}°, {np.degrees(el[0]):.2f}°")
            
            # Check unit vector normalization
            norm = np.sqrt(Eu[0]**2 + Nu[0]**2 + Uu[0]**2)
            print(f"    Unit vector norm: {norm:.6f} (should be 1.0)")
    
    print("\n" + "="*70)
    print("TEST 2: XY → Geo → XY roundtrip")
    print("-"*70)
    
    for height in [100000]:  # Test at 100km
        print(f"\nHeight: {height/1000:.0f} km")
        for name, x, y in test_points:
            print(f"\n  {name}: ({x:.1f}, {y:.1f})")
            
            # XY to Geo
            lat_arr, lon_arr, h_arr = xyToGeoPP_iter(
                np.array([x]), np.array([y]), np.array([height]), pp.JD, pp
            )
            lat = lat_arr[0]
            lon = lon_arr[0]
            h = h_arr[0]
            
            print(f"    → Geo: Lat={lat:.6f}°, Lon={lon:.6f}°, H={h:.1f}m")
            
            # Geo back to XY
            x_arr, y_arr = geoToXYPP_iter(
                np.array([lat]), np.array([lon]), np.array([h]), pp.JD, pp
            )
            x_back = x_arr[0]
            y_back = y_arr[0]
            
            print(f"    → XY: ({x_back:.3f}, {y_back:.3f})")
            
            error = np.sqrt((x_back - x)**2 + (y_back - y)**2)
            print(f"    Error: {error:.3f} px")
            
            if error > 1.0:
                print(f"    ⚠️ Large error!")
    
    print("\n" + "="*70)
    print("TEST 3: Geo → XY → Geo roundtrip")
    print("-"*70)
    
    # Test with some geodetic coordinates
    test_geo_points = [
        ("Near station", pp.lat, pp.lon + 0.5, 100000),  # 0.5° east
        ("North of station", pp.lat + 0.5, pp.lon, 100000),  # 0.5° north
        ("Northeast", pp.lat + 0.3, pp.lon + 0.3, 100000),
    ]
    
    for name, lat, lon, h in test_geo_points:
        print(f"\n  {name}: Lat={lat:.6f}°, Lon={lon:.6f}°, H={h/1000:.0f}km")
        
        # Geo to XY
        x_arr, y_arr = geoToXYPP_iter(
            np.array([lat]), np.array([lon]), np.array([h]), pp.JD, pp
        )
        
        # Check if point is in image
        if len(x_arr) > 0 and 0 <= x_arr[0] <= pp.X_res and 0 <= y_arr[0] <= pp.Y_res:
            x = x_arr[0]
            y = y_arr[0]
            print(f"    → XY: ({x:.3f}, {y:.3f})")
            
            # XY back to Geo
            lat_arr, lon_arr, h_arr = xyToGeoPP_iter(
                np.array([x]), np.array([y]), np.array([h]), pp.JD, pp
            )
            lat_back = lat_arr[0]
            lon_back = lon_arr[0]
            h_back = h_arr[0]
            
            print(f"    → Geo: Lat={lat_back:.6f}°, Lon={lon_back:.6f}°, H={h_back:.1f}m")
            
            # Compute error in meters
            xyz1 = geo2Cartesian(lat, lon, h, pp.JD)
            xyz2 = geo2Cartesian(lat_back, lon_back, h_back, pp.JD)
            error_m = np.sqrt(np.sum((np.array(xyz1) - np.array(xyz2))**2))
            
            print(f"    Error: {error_m:.2f} m")
            
            if error_m > 100:
                print(f"    ⚠️ Large error!")
        else:
            print(f"    Point outside image FOV")

if __name__ == "__main__":
    test_enu_functions()