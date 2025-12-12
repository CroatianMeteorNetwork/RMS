
import os
import time
import datetime
from typing import List, Tuple, Optional

import numpy as np
try:
    from skyfield.api import load, Topos, EarthSatellite, wgs84, Loader
    from skyfield.sgp4lib import EarthSatellite
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Formats.Platepar import Platepar
from RMS.Routines.SphericalPolygonCheck import sphericalPolygonCheck
from RMS.Misc import getRmsRootDir
from RMS.Astrometry.Conversions import datetime2JD
import urllib.request
import traceback
import argparse


def loadTLEs(cache_dir, cache_file_name="active.txt", 
                         url="http://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle", 
                         max_age_hours=24.0):
    """ Load TLEs from a local cache or download them if the cache is old or missing.

    Arguments:
        cache_dir: [str] Directory to store the TLE file.

    Keyword arguments:
        cache_file_name: [str] Name of the cache file. "active.txt" by default.
        url: [str] URL to download TLEs from.
        max_age_hours: [float] Maximum age of the cache in hours. 24.0 by default.

    Return:
        satellites: [list] List of properties for EarthSatellite objects.
    """
    if not SKYFIELD_AVAILABLE:
        print("Skyfield not available, cannot load TLEs.")
        return []

    cache_path = os.path.join(cache_dir, cache_file_name)
    
    download = True
    if os.path.exists(cache_path):
        # Check file age
        file_age_hours = (time.time() - os.path.getmtime(cache_path))/3600.0
        if file_age_hours < max_age_hours:
            download = False
            # print(f"Using cached TLEs from {cache_path} ({file_age_hours:.1f} hours old).")
    
    if download:
        print(f"Downloading TLEs from {url}...")
        try:
            # Download the TLE file
            os.makedirs(cache_dir, exist_ok=True)
            urllib.request.urlretrieve(url, cache_path)
            print(f"Downloaded TLEs to {cache_path}")
            
        except Exception as e:
            print(f"Failed to download TLEs: {e}")
            if os.path.exists(cache_path):
                print("Falling back to existing (old) cache.")
            else:
                return []

    # Load satellites from file
    try:
        satellites = load.tle_file(cache_path)
        return satellites
    except Exception as e:
        print(f"Error loading TLEs from {cache_path}: {e}")
        return []


class SatellitePredictor:
    def __init__(self, lat: float, lon: float, elev: float, time_begin: datetime.datetime, time_end: datetime.datetime):
        """
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            elev: Elevation in meters.
            time_begin: Start of the observation.
            time_end: End of the observation.
        """
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.time_begin = time_begin
        self.time_end = time_end
        
        self.ts = load.timescale()
        self.observer = wgs84.latlon(lat, lon, elevation_m=elev)
        self.t0 = self.ts.from_datetime(time_begin.replace(tzinfo=datetime.timezone.utc))
        self.t1 = self.ts.from_datetime(time_end.replace(tzinfo=datetime.timezone.utc))


    def getSatelliteTracks(self, platepar, fov_polygon, satellites, time_step_seconds=1.0):
        """ Get tracks of satellites visible in the FOV.

        Arguments:
            platepar: [Platepar instance] Platepar object for projection.
            fov_polygon: [list] List of (RA, Dec) tuples defining the FOV in degrees.
            satellites: [list] List of EarthSatellite objects.

        Keyword arguments:
            time_step_seconds: [float] Time step for track points. 1.0 by default.

        Return:
            visible_tracks: [list] List of dictionaries suitable for plotting:
                [
                    {
                        "name": str,
                        "x": np.array,
                        "y": np.array
                    }, 
                    ...
                ]
        """
        
        # --- DEBUG PRINTS ---
        print("\n--- SatellitePredictor Debug Info ---")
        print(f"Time Begin: {self.time_begin} (UTC)")
        print(f"Time End:   {self.time_end} (UTC)")
        print(f"Duration:   {(self.time_end - self.time_begin).total_seconds()}s")
        print(f"Observer:   Lat={self.lat:.5f}, Lon={self.lon:.5f}, Elev={self.elev:.1f}m")
        print(f"Platepar:   Res={platepar.X_res}x{platepar.Y_res}, RA={platepar.RA_d:.5f}, Dec={platepar.dec_d:.5f}")
        print(f"FOV Poly:   {len(fov_polygon)} points")
        if fov_polygon:
            ras = [p[0] for p in fov_polygon]
            decs = [p[1] for p in fov_polygon]
            print(f"FOV RA Range:  {min(ras):.4f} to {max(ras):.4f}")
            print(f"FOV Dec Range: {min(decs):.4f} to {max(decs):.4f}")
        print(f"Satellites: {len(satellites)} to check")
        print("-------------------------------------\n")
        # --------------------
        
        # Determine time points
        duration = (self.time_end - self.time_begin).total_seconds()
        if duration <= 0:
            return []
            
        # Compute time steps
        steps = int(duration/time_step_seconds) + 1
        if steps > 1:
             time_list = [
                (self.time_begin + datetime.timedelta(seconds=i*time_step_seconds)).replace(tzinfo=datetime.timezone.utc)
                for i in range(steps)
            ]
             times = self.ts.from_datetimes(time_list)
        else:
             times = self.ts.from_datetime(self.time_begin.replace(tzinfo=datetime.timezone.utc))

        
        visible_tracks = []
        
        # Filter satellites by checking their elevation at the midpoint of the time range
        # This significantly reduces the number of full propagations needed
        t_mid = self.ts.from_datetime((self.time_begin + (self.time_end - self.time_begin)/2).replace(tzinfo=datetime.timezone.utc))
        print(f"Checking {len(satellites)} satellites...")
        
        valid_sats = []
        
        # Loop through satellites and check if they are above the horizon at t_mid
        for sat in satellites:
            try:
                # Quick check at mid time
                orbit_quality = ((sat - self.observer).at(t_mid).altaz()[0].degrees > -5)
                if orbit_quality:
                    valid_sats.append(sat)
            except Exception:
                continue
                
        # print(f"Found {len(valid_sats)} candidates above horizon.")
        
        for sat in valid_sats:
            try:
                # Propagate for all times
                topocentric = (sat - self.observer).at(times)
                ra, dec, distance = topocentric.radec()
                
                ra_degs = ra._degrees
                dec_degs = dec._degrees
                
                # Check for Earth shadow using the DE421 ephemeris
                if not hasattr(self, 'eph'):
                    
                     # Use skyfield Loader to manage cache and downloads
                     cache_dir = os.path.join(getRmsRootDir(), ".skyfield_cache")
                     os.makedirs(cache_dir, exist_ok=True)
                     loader = Loader(cache_dir)
                     
                     try:
                        self.eph = loader('de421.bsp')
                     except Exception as e:
                        print(f"Could not load de421.bsp: {e}. Shadow check disabled.")
                        self.eph = None
                
                sunlit = np.ones(len(ra_degs), dtype=bool)
                if self.eph:
                     try:
                         sunlit = topocentric.is_sunlit(self.eph)
                     except Exception as e:
                         # Fallback if download failed or some other issue
                         pass
                
                # Filter points
                # Let's keep indices where sunlit is True
                if not np.any(sunlit):
                     continue # Entirely in shadow
                
                # Filter coordinates to keep only sunlit points
                ra_degs = ra_degs[sunlit]
                dec_degs = dec_degs[sunlit]
                jds_utc_subset = times.ut1[sunlit] 
                
                # Check if any point is in FOV. Only check points inside image bounds by projecting
                # the whole track segment.
                
                # Wrap RA
                ra_degs = ra_degs%360
                
                # Generate list of points
                test_points = np.column_stack((ra_degs, dec_degs))
                
                # Use SphericalPolygonCheck to check which points are inside the FOV
                inside = sphericalPolygonCheck(fov_polygon, test_points)
                
                if np.any(inside):
                    # Project RA/Dec to per-pixel XY
                    
                    # We need the JD for the astrometry (time dependent? Only if alt/az is used or refraction?)
                    # raDecToXYPP(ra, dec, jd, platepar)
                    
                    jds_utc = times.ut1
                    
                    x_all = []
                    y_all = []
                    ra_all = []
                    dec_all = []
                    
                    # Compute XY coordinates for all points
                    for r, d, jd in zip(ra_degs, dec_degs, jds_utc_subset):
                        x, y = raDecToXYPP(np.array([float(r)]), np.array([float(d)]), float(jd), platepar)
                        x_all.append(x[0])
                        y_all.append(y[0])
                        ra_all.append(r)
                        dec_all.append(d)
                        
                    # Filter out ridiculous values (behind camera etc) if any
                    x_all = np.array(x_all)
                    y_all = np.array(y_all)
                    ra_all = np.array(ra_all)
                    dec_all = np.array(dec_all)
                    
                    # Check if valid
                    # Simple check: inside image bounds?
                    # w = platepar.X_res
                    # h = platepar.Y_res
                    
                    # If at least some points are within reasonable bounds
                    if len(x_all) > 0:
                        visible_tracks.append({
                            "name": sat.name,
                            "x": x_all,
                            "y": y_all,
                            "ra": ra_all,
                            "dec": dec_all
                        })
                        
            except Exception as e:
                print(f"Error propagating {sat.name}: {e}")
                traceback.print_exc()
                continue
                
        return visible_tracks

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict satellite tracks in FOV.")
    parser.add_argument("platepar_path", type=str, help="Path to platepar file.")
    parser.add_argument("--time_start", type=str, help="Start time (YYYYMMDD_HHMMSS or ISO). Defaults to now.")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (default 60).")
    parser.add_argument("--tle_cache", type=str, default=None, help="Directory for TLE cache.")
    parser.add_argument("--tle_file", type=str, default=None, help="Path to a specific TLE file to use (skips download).")
    parser.add_argument("--show-plots", action="store_true", help="Display a plot of the satellite tracks.")

    args = parser.parse_args()

    # Load platepar
    if not os.path.exists(args.platepar_path):
        print(f"Error: Platepar file not found: {args.platepar_path}")
        exit(1)

    pp = Platepar()
    pp.read(args.platepar_path, use_flat=False)
    
    # Time handling
    if args.time_start:
        try:
            t_start = datetime.datetime.strptime(args.time_start, "%Y%m%d_%H%M%S")
        except ValueError:
             try:
                t_start = datetime.datetime.fromisoformat(args.time_start)
             except ValueError:
                 print("Invalid time format. Use YYYYMMDD_HHMMSS or ISO format.")
                 exit(1)
    else:
        t_start = datetime.datetime.utcnow()

    # Ensure UTC
    if t_start.tzinfo is None:
        t_start = t_start.replace(tzinfo=datetime.timezone.utc)
        
    t_end = t_start + datetime.timedelta(seconds=args.duration)

    print(f"Time Range: {t_start} to {t_end}")
    print(f"Location: Lat={pp.lat:.4f}, Lon={pp.lon:.4f}, Elev={pp.elev:.1f}m")

    # Load TLEs
    if args.tle_file:
         if not os.path.exists(args.tle_file):
              print(f"Error: TLE file not found: {args.tle_file}")
              exit(1)
         print(f"Loading TLEs from {args.tle_file}...")
         try:
             sats = load.tle_file(args.tle_file)
         except Exception as e:
             print(f"Error loading TLE file: {e}")
             exit(1)
    else:
        if args.tle_cache:
             cache_dir = args.tle_cache
        else:
             cache_dir = os.path.join(getRmsRootDir(), ".skyfield_cache")

        os.makedirs(cache_dir, exist_ok=True)
        sats = loadTLEs(cache_dir, max_age_hours=24)
        
    if not sats:
        print("No TLEs loaded.")
        exit(1)

    predictor = SatellitePredictor(pp.lat, pp.lon, pp.elev, t_start, t_end)

    # Compute FOV Polygon (sampling edges)
    w = pp.X_res
    h = pp.Y_res
    
    # Define edges
    edges = [
        ((0, 0), (w, 0)),
        ((w, 0), (w, h)),
        ((w, h), (0, h)),
        ((0, h), (0, 0))
    ]
    
    fov_poly = []
    samples_per_side = 10
    
    jd = datetime2JD(t_start)
    
    for (x_start, y_start), (x_end, y_end) in edges:
        xs = np.linspace(x_start, x_end, samples_per_side, endpoint=False)
        ys = np.linspace(y_start, y_end, samples_per_side, endpoint=False)
        
        n = len(xs)
        jd_arr = [jd]*n
        level_arr = [1]*n
        
        _, r_arr, d_arr, _ = xyToRaDecPP(jd_arr, xs, ys, level_arr, pp, jd_time=True, extinction_correction=False)
        
        for r, d in zip(r_arr, d_arr):
            fov_poly.append((r, d))

    tracks = predictor.getSatelliteTracks(pp, fov_poly, sats)


    print(f"\nFound {len(tracks)} visible satellites:")
    print("-" * 60)
    for track in tracks:
        name = track['name']
        x = track['x']
        y = track['y']
        ra = track['ra']
        dec = track['dec']
        
        # Get begin and end points
        x1, x2 = x[0], x[-1]
        y1, y2 = y[0], y[-1]
        r1, r2 = ra[0], ra[-1]
        d1, d2 = dec[0], dec[-1]
        
        print(f"{name}")
        print(f"      {'begin':>10}, {'end':>10}")
        print(f"ra    = {r1:10.4f}, {r2:10.4f}")
        print(f"dec   = {d1:10.4f}, {d2:10.4f}")
        print(f"x     = {x1:10.2f}, {x2:10.2f}")
        print(f"y     = {y1:10.2f}, {y2:10.2f}")
        print("-" * 60)

    if args.show_plots and tracks:
        print("Plotting tracks...")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Error: matplotlib is required for plotting. Please install it.")
            exit(1)

        plt.figure(figsize=(10, 8))
        plt.title(f"Satellite Tracks ({len(tracks)} visible)")
        
        # Plot FOV bounds
        plt.plot([0, w, w, 0, 0], [0, 0, h, h, 0], 'k--', linewidth=2, label='FOV')

        for track in tracks:
            x = track['x']
            y = track['y']
            name = track['name']
            
            p = plt.plot(x, y, label=name)
            color = p[0].get_color()
            
            # Find a point inside the image for the label
            label_x, label_y = x[0], y[0]
            for xi, yi in zip(x, y):
                 if 0 <= xi <= w and 0 <= yi <= h:
                      label_x, label_y = xi, yi
                      break
            
            # Label
            plt.text(label_x, label_y, name, fontsize=8, color=color, clip_on=True)

        plt.xlim(0, w)
        plt.ylim(h, 0) # Invert Y axis for image coords
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.grid(True, alpha=0.3)
        plt.show()
