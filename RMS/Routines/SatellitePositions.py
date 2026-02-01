
import os
import time
import datetime

import numpy as np
try:
    from skyfield.api import load, wgs84, Loader
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
import re
import glob

import tempfile


def findClosestTLEFile(directory_path, target_time):
    """
    Scans the given directory for TLE files with the format TLE_YYYYMMDD_HHMMSS_...
    and finds the one closest to the target_time.
    
    Arguments:
        directory_path: [str] Path to the directory containing TLE files.
        target_time: [datetime] The time for which we want the closest TLEs.
        
    Returns:
        [str] Path to the closest TLE file, or None if no suitable file found or "current time" is closer (indicating download is preferred).
    """

    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return None

    print(f"Scanning TLE directory: {directory_path}")

    files = glob.glob(os.path.join(directory_path, "TLE_*.txt"))
    
    best_file = None
    min_diff = None
    
    # Ensure target_time is UTC
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=datetime.timezone.utc)
        
    # Current time for comparison (downloaded TLEs)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    diff_now = abs((now_utc - target_time).total_seconds())
    
    print(f"Time difference to NOW (downloaded TLEs): {diff_now/3600:.2f} hours")
    
    for f in files:
        basename = os.path.basename(f)
        # Regex to capture the first timestamp group: YYYYMMDD_HHMMSS
        match = re.search(r"TLE_(\d{8}_\d{6})_", basename)
        if match:
            ts_str = match.group(1)
            try:
                # Parse timestamp, assuming UTC
                dt = datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                dt = dt.replace(tzinfo=datetime.timezone.utc)
                
                diff = abs((dt - target_time).total_seconds())
                
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    best_file = f
            except ValueError:
                continue

    if best_file:
        print(f"Best file found: {os.path.basename(best_file)} (diff: {min_diff/3600:.2f} hours)")
        # If "now" is closer than the best file, return None to signal "use downloaded"
        if diff_now < min_diff:
            print("Current time is closer than any file. Using downloaded TLEs.")
            return None
        else:
            return best_file
    else:
        print("No matching TLE files found in directory.")
        return None


def loadRobustTLEs(file_path):
    """ Loads TLEs from a file, handling potential errors with non-standard IDs (e.g. 'T0000').
    
    Arguments:
        file_path: [str] Path to the TLE file.
        
    Return:
        satellites: [list] List of properties for EarthSatellite objects.
    """
    
    try:
        # Try loading normally first
        return load.tle_file(file_path)

    except ValueError as e:
        # Check if it's the specific int conversion error
        if "invalid literal for int()" not in str(e):
            raise e
            
        print(f"Warning: Standard TLE loading failed ({e}). Attempting robust loading with sanitized IDs...")
        
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Sanitize IDs
        # We need to replace non-numeric IDs in columns 2-7 with a numeric dummy ID
        # We must ensure line 1 and 2 for the same sat get the same ID.
        
        sanitized_lines = []
        
        # Map original bad ID -> new dummy ID
        # Start from 90000 to avoid conflicts with common sats
        next_dummy_id = 90000 
        id_map = {}
        
        for line in lines:
            # Check if line 1 or 2 of TLE
            if len(line) > 60 and line[0] in ('1', '2') and line[1] == ' ':
                # Extract ID
                curr_id = line[2:7]
                
                # Check if it's alphanumeric
                if not curr_id.strip().isdigit():
                    # It's bad. Do we have a replacement?
                    if curr_id not in id_map:
                        id_map[curr_id] = f"{next_dummy_id:05d}"
                        next_dummy_id += 1
                        
                    new_id = id_map[curr_id]
                    # Replace in line
                    line = line[:2] + new_id + line[7:]
            
            sanitized_lines.append(line)
            
        # Write to temp file and load
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.writelines(sanitized_lines)
                tmp_path = tmp.name
            
            sats = load.tle_file(tmp_path)
            # Restore names/original IDs? 
            # The EarthSatellite object has .model.satnum (integer).
            # The name is separate.
            # We can't easily restore the textual string ID inside the object if it stores int.
            # But functionality should be fine.
            print(f"Successfully loaded {len(sats)} satellites after sanitization.")
            return sats
             
        finally:
             if tmp_path and os.path.exists(tmp_path):
                 os.remove(tmp_path)


def loadTLEs(cache_dir, cache_file_name="active.txt", 
                         url="http://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle", 
                         max_age_hours=24.0,
                         use_daily_cache=True):
    """ Load TLEs from a local cache or download them if the cache is old or missing.
    
    By default, uses daily cache files with date-stamped names (TLE_YYYYMMDD_HHMMSS_<name>.txt).
    This allows tracking of TLE updates over time and enables findClosestTLEFile to select
    the most appropriate TLE data for a given observation time.

    Arguments:
        cache_dir: [str] Directory to store the TLE file.

    Keyword arguments:
        cache_file_name: [str] Base name of the cache file. "active.txt" by default.
        url: [str] URL to download TLEs from.
        max_age_hours: [float] Maximum age of the cache in hours. 24.0 by default.
        use_daily_cache: [bool] If True, creates date-stamped cache files. If False, uses 
                                the legacy single-file cache behavior. True by default.

    Return:
        satellites: [list] List of properties for EarthSatellite objects.
    """
    if not SKYFIELD_AVAILABLE:
        print("Skyfield not available, cannot load TLEs.")
        return []

    # Generate cache file path with optional daily naming
    if use_daily_cache:
        # Create date-stamped filename: TLE_YYYYMMDD_HHMMSS_<original_name>
        now = datetime.datetime.utcnow()
        date_str = now.strftime("%Y%m%d")
        
        # Check if today's cache already exists
        existing_files = []
        if os.path.exists(cache_dir):
            pattern = f"TLE_{date_str}_*_{cache_file_name}"
            existing_files = glob.glob(os.path.join(cache_dir, pattern))
            
        # Use the most recent file from today if it exists and is fresh
        cache_path = None
        download = True
        
        if existing_files:
            # Sort by modification time, newest first
            existing_files.sort(key=os.path.getmtime, reverse=True)
            newest_file = existing_files[0]
            
            # Check file age
            file_age_hours = (time.time() - os.path.getmtime(newest_file))/3600.0
            if file_age_hours < max_age_hours:
                cache_path = newest_file
                download = False
                # print(f"Using cached TLEs from {cache_path} ({file_age_hours:.1f} hours old).")
        
        # If we need to download, create a new timestamped file
        if download:
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            daily_cache_name = f"TLE_{timestamp_str}_{cache_file_name}"
            cache_path = os.path.join(cache_dir, daily_cache_name)
    else:
        # Legacy behavior: single cache file
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
        satellites = loadRobustTLEs(cache_path)
        return satellites
    except Exception as e:
        print(f"Error loading TLEs from {cache_path}: {e}")
        return []


class SatellitePredictor:
    
    # Cache for ephemeris and timescale to avoid reloading
    _eph = None
    _ts = None

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
        
        # Load timescale if not already loaded
        if SatellitePredictor._ts is None:
             SatellitePredictor._ts = load.timescale()
             
        self.ts = SatellitePredictor._ts
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
                # Check if the satellite is above the horizon (-5 deg to account for atmospheric refraction) at t_mid
                above_horizon = ((sat - self.observer).at(t_mid).altaz()[0].degrees > -5)
                if above_horizon:
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
                if SatellitePredictor._eph is None:
                    
                     # Use skyfield Loader to manage cache and downloads
                     cache_dir = os.path.join(getRmsRootDir(), ".skyfield_cache")
                     os.makedirs(cache_dir, exist_ok=True)
                     loader = Loader(cache_dir)
                     
                     try:
                        SatellitePredictor._eph = loader('de421.bsp')
                     except Exception as e:
                        print(f"Could not load de421.bsp: {e}. Shadow check disabled.")
                        # use False to indicate "tried and failed"
                        SatellitePredictor._eph = False

                self.eph = SatellitePredictor._eph
                
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
                    # Filter points to only those truly inside the spherical FOV
                    ra_degs = ra_degs[inside]
                    dec_degs = dec_degs[inside]
                    jds_utc_subset = jds_utc_subset[inside]

                    # Project RA/Dec to per-pixel XY
                    
                    # Compute XY coordinates for all points
                    # Use the first timestamp for all points as raDecToXYPP expects a scalar JD
                    x_all, y_all = raDecToXYPP(ra_degs, dec_degs, float(jds_utc_subset[0]), platepar)
                    ra_all = ra_degs
                    dec_all = dec_degs
                    

                    
                    # Check if at least 2 points are inside the image bounds
                    w = platepar.X_res
                    h = platepar.Y_res
                    
                    # Create a boolean mask for points inside the image
                    inside_mask = (x_all >= 0) & (x_all <= w) & (y_all >= 0) & (y_all <= h)
                    


                    if np.sum(inside_mask) >= 2:
                        # Clip the track to only include points inside the FOV
                        time_clipped = jds_utc_subset[inside_mask]
                        x_clipped = x_all[inside_mask]
                        y_clipped = y_all[inside_mask]
                        ra_clipped = ra_all[inside_mask]
                        dec_clipped = dec_all[inside_mask]

                        visible_tracks.append({
                            "name": sat.name,
                            "time": time_clipped,
                            "x": x_clipped,
                            "y": y_clipped,
                            "ra": ra_clipped,
                            "dec": dec_clipped
                            
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
    parser.add_argument("--tle_file", type=str, default=None, help="Path to a specific TLE file (or directory containing TLE files) to use (skips download).")
    parser.add_argument("--show-plots", action="store_true", help="Display a plot of the satellite tracks.")

    args = parser.parse_args()

    if not SKYFIELD_AVAILABLE:
        print("Error: The 'skyfield' library is required to run this script. Please install it using 'pip install skyfield'.")
        exit(1)

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
    sats = []
    tle_file_path = None

    # If a TLE file/directory is specified, try to find the best file
    if args.tle_file:
         if os.path.isdir(args.tle_file):
             print(f"Searching for closest TLE file in {args.tle_file}...")
             tle_file_path = findClosestTLEFile(args.tle_file, t_start)
         elif os.path.exists(args.tle_file):
             tle_file_path = args.tle_file
         else:
             print(f"Error: TLE file not found: {args.tle_file}")
             exit(1)

    # If we found a file, load it
    if tle_file_path:
        print(f"Loading TLEs from {tle_file_path}...")
        try:
            sats = load.tle_file(tle_file_path)
        except Exception as e:
            print(f"Error loading TLE file: {e}")
            exit(1)
    
    # If no file was specified or found (and we didn't exit), try loading/downloading from cache
    if not sats:
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
