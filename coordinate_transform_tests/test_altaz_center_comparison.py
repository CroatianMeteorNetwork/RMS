#!/usr/bin/env python3
"""
Compare the original platepar alt/az center values with recomputed ones.
"""

import sys
import os
import numpy as np

# Add RMS to path
sys.path.insert(0, '/home/luc/source/RMS')

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.Conversions import jd2Date
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz

def test_center_values():
    """Compare platepar center values."""
    
    # Load platepar
    pp = Platepar()
    pp.read('/home/luc/data/advect_staging/US0001/US0001_20250707_031251_to_20250707_111038/platepar_cmn2010.cal')
    
    print("ALT/AZ CENTER COMPARISON")
    print("="*70)
    print(f"Platepar RA center: {pp.RA_d:.6f}°")
    print(f"Platepar Dec center: {pp.dec_d:.6f}°")
    print(f"Platepar Alt center: {pp.alt_centre:.6f}°")
    print(f"Platepar Az center: {pp.az_centre:.6f}°")
    print(f"JD: {pp.JD:.6f}")
    print(f"Lat: {pp.lat:.6f}°, Lon: {pp.lon:.6f}°")
    print(f"Refraction setting: {pp.refraction}")
    
    print("\n" + "-"*70)
    print("Recomputing Alt/Az from RA/Dec with refraction=True...")
    az_refr, alt_refr = cyTrueRaDec2ApparentAltAz(
        np.radians(pp.RA_d), np.radians(pp.dec_d),
        pp.JD, np.radians(pp.lat), np.radians(pp.lon),
        True  # refraction=True
    )
    az_refr = np.degrees(az_refr)
    alt_refr = np.degrees(alt_refr)
    
    print(f"With refraction - Alt: {alt_refr:.6f}°, Az: {az_refr:.6f}°")
    print(f"Δ Alt: {alt_refr - pp.alt_centre:.6f}°")
    print(f"Δ Az: {az_refr - pp.az_centre:.6f}°")
    
    print("\n" + "-"*70)
    print("Recomputing Alt/Az from RA/Dec with refraction=False...")
    az_norefr, alt_norefr = cyTrueRaDec2ApparentAltAz(
        np.radians(pp.RA_d), np.radians(pp.dec_d),
        pp.JD, np.radians(pp.lat), np.radians(pp.lon),
        False  # refraction=False
    )
    az_norefr = np.degrees(az_norefr)
    alt_norefr = np.degrees(alt_norefr)
    
    print(f"No refraction - Alt: {alt_norefr:.6f}°, Az: {az_norefr:.6f}°")
    print(f"Δ Alt: {alt_norefr - pp.alt_centre:.6f}°")
    print(f"Δ Az: {az_norefr - pp.az_centre:.6f}°")
    
    print("\n" + "-"*70)
    print("Using platepar's refraction setting...")
    az_pp, alt_pp = cyTrueRaDec2ApparentAltAz(
        np.radians(pp.RA_d), np.radians(pp.dec_d),
        pp.JD, np.radians(pp.lat), np.radians(pp.lon),
        pp.refraction  # Use platepar's setting
    )
    az_pp = np.degrees(az_pp)
    alt_pp = np.degrees(alt_pp)
    
    print(f"Platepar refraction={pp.refraction} - Alt: {alt_pp:.6f}°, Az: {az_pp:.6f}°")
    print(f"Δ Alt: {alt_pp - pp.alt_centre:.6f}°")
    print(f"Δ Az: {az_pp - pp.az_centre:.6f}°")

if __name__ == "__main__":
    test_center_values()