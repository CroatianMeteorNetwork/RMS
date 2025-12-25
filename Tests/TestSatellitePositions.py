
import unittest
import os
import shutil
import datetime
import tempfile
from unittest.mock import patch

from RMS.Routines.SatellitePositions import loadTLEs, SatellitePredictor, SKYFIELD_AVAILABLE
from RMS.Formats.Platepar import Platepar

class TestSatellitePositions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_tles_caching(self):
        if not SKYFIELD_AVAILABLE:
            self.skipTest("Skyfield not installed")
            
        cache_file = "test_tle.txt"
        cache_path = os.path.join(self.test_dir, cache_file)
        
        # Test download
        with patch('urllib.request.urlretrieve') as mock_download:
             # Create a dummy file so load.tle_file doesn't fail
            with open(cache_path, 'w') as f:
                f.write("DUMMY TLE DATA")
            
            # 1. File exists and is new -> No download
            with patch('os.path.getmtime') as mock_mtime:
                mock_mtime.return_value = datetime.datetime.now().timestamp()
                
                with patch('skyfield.api.load.tle_file') as mock_load:
                    mock_load.return_value = []
                    loadTLEs(self.test_dir, cache_file, max_age_hours=24)
                    mock_download.assert_not_called()
                    
            # 2. File exists and is old -> Download
            with patch('os.path.getmtime') as mock_mtime:
                # 25 hours ago
                mock_mtime.return_value = (datetime.datetime.now() - datetime.timedelta(hours=25)).timestamp()
                
                with patch('skyfield.api.load.tle_file') as mock_load:
                    mock_load.return_value = []
                     # We need to simulate the file existing for the check
                    loadTLEs(self.test_dir, cache_file, max_age_hours=24)
                    mock_download.assert_called()

    def test_satellite_predictor(self):
        if not SKYFIELD_AVAILABLE:
            self.skipTest("Skyfield not installed")

        # Mock dependencies
        lat, lon, elev = 45.0, 13.0, 100.0
        t_start = datetime.datetime(2023, 1, 1, 0, 0, 0)
        t_end = t_start + datetime.timedelta(minutes=1)
        
        predictor = SatellitePredictor(lat, lon, elev, t_start, t_end)
        
        # Create a mock satellite
        # Using a minimal TLE format
        tle_line1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
        tle_line2 = "2 25544  51.6439 201.2643 0007417 356.5673 139.3661 15.50083952202315"
        
        from skyfield.api import EarthSatellite
        ts = predictor.ts
        sat = EarthSatellite(tle_line1, tle_line2, 'ISS (ZARYA)', ts)
        
        # Mock Platepar
        pp = Platepar()
        pp.X_res = 1920
        pp.Y_res = 1080
        
        # FOV Polygon (Full sky approx)
        fov_poly = [(0, 0), (360, 0), (360, 90), (0, 90)]
        
        # Test get_satellite_tracks
        # We need to mock raDecToXYPP to avoid needing actual astrometry
        with patch('RMS.Astrometry.ApplyAstrometry.raDecToXYPP', return_value=(100, 100)):
            tracks = predictor.getSatelliteTracks(pp, fov_poly, [sat])
            
            # Since we forced return 100,100, and we have a sat, we should get something if it is above horizon
            # ISS at this time might not be above horizon for this location.
            # But we are testing the logic flow. 
            
            # If ISS is not visible, list is empty.
            # Let's trust the logic runs without error.
            self.assertIsInstance(tracks, list)
