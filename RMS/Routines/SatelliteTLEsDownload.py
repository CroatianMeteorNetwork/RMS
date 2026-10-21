#!/usr/bin/env python
"""
Standalone script to download and cache satellite TLEs.

This script downloads current TLE (Two-Line Element) data for satellites and stores them
in a date-stamped cache file. It's designed to be run daily (e.g., via cron job) to maintain
a historical archive of TLE data over time.

The cache files are named with the format: TLE_YYYYMMDD_HHMMSS_<cache_name>.txt

Example usage:
    # Download using default settings (to ~/.skyfield_cache)
    python -m RMS.Routines.SatelliteTLEsDownload
    
    # Download to a custom directory
    python -m RMS.Routines.SatelliteTLEsDownload --cache-dir /path/to/custom/cache
    
    # Force re-download even if today's cache exists
    python -m RMS.Routines.SatelliteTLEsDownload --force
    
    # Download from a custom URL
    python -m RMS.Routines.SatelliteTLEsDownload --url "http://example.com/tle_data.txt"
"""

import os
import sys
import argparse

from RMS.Routines.SatellitePositions import loadTLEs, SKYFIELD_AVAILABLE
from RMS.Misc import getRmsRootDir


def main():
    """Main function for TLE download script."""
    
    parser = argparse.ArgumentParser(
        description="Download and cache satellite TLEs with daily timestamping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --cache-dir /path/to/custom/cache
  %(prog)s --force
  %(prog)s --url "http://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
        """
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store cached TLE files. Defaults to RMS root .skyfield_cache directory."
    )
    
    parser.add_argument(
        "--cache-name",
        type=str,
        default="active.txt",
        help="Base name for the cache file (default: active.txt). The actual filename will be "
             "TLE_YYYYMMDD_HHMMSS_<cache-name>."
    )
    
    parser.add_argument(
        "--url",
        type=str,
        default="http://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
        help="URL to download TLEs from. Defaults to Celestrak active satellites."
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if today's cache already exists and is fresh."
    )
    
    args = parser.parse_args()
    
    # Check if skyfield is available
    if not SKYFIELD_AVAILABLE:
        print("ERROR: The 'skyfield' library is required to download TLEs.")
        print("Please install it using: pip install skyfield")
        return 1
    
    # Determine cache directory
    if args.cache_dir:
        cache_dir = os.path.abspath(args.cache_dir)
    else:
        cache_dir = os.path.join(getRmsRootDir(), ".skyfield_cache")
    
    print("="*70)
    print("Satellite TLE Download Script")
    print("="*70)
    print(f"Cache directory: {cache_dir}")
    print(f"Cache base name: {args.cache_name}")
    print(f"TLE source URL:  {args.url}")
    print(f"Force download:  {args.force}")
    print("="*70)
    print()
    
    # Set max_age_hours to 0 if force is enabled (this will force a download)
    max_age_hours = 0.0 if args.force else 24.0
    
    # Download and cache TLEs
    try:
        satellites = loadTLEs(
            cache_dir=cache_dir,
            cache_file_name=args.cache_name,
            url=args.url,
            max_age_hours=max_age_hours,
            use_daily_cache=True
        )
        
        if satellites:
            print()
            print("="*70)
            print(f"SUCCESS: Loaded {len(satellites)} satellites from TLE cache.")
            print("="*70)
            return 0
        else:
            print()
            print("="*70)
            print("WARNING: No satellites were loaded. Check the error messages above.")
            print("="*70)
            return 1
            
    except Exception as e:
        print()
        print("="*70)
        print(f"ERROR: Failed to download TLEs: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
