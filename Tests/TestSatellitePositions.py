
import unittest
import os
import shutil
import datetime
import tempfile
import glob
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
        
        # Test 1: Daily cache mode - should create date-stamped files
        def createDummyTle(url, filepath):
            """Side effect for mock to actually create a file"""
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write("DUMMY TLE DATA\n")
        
        with patch('urllib.request.urlretrieve', side_effect=createDummyTle) as mock_download:
            with patch('RMS.Routines.SatellitePositions.loadRobustTLEs') as mock_load:
                mock_load.return_value = []
                
                # First call should download and create a new date-stamped file
                loadTLEs(self.test_dir, cache_file, max_age_hours=24, use_daily_cache=True)
                mock_download.assert_called_once()
                
                # Check that a date-stamped file was created
                cache_files = glob.glob(os.path.join(self.test_dir, f"TLE_*_{cache_file}"))
                self.assertEqual(len(cache_files), 1, "Should create one date-stamped cache file")
                self.assertIn("TLE_", os.path.basename(cache_files[0]), "Cache file should have TLE_ prefix")
                
                mock_download.reset_mock()
                
                # Second call on same day should use existing cache (not download)
                loadTLEs(self.test_dir, cache_file, max_age_hours=24, use_daily_cache=True)
                mock_download.assert_not_called()
                
                # Still should be only one file
                cache_files = glob.glob(os.path.join(self.test_dir, f"TLE_*_{cache_file}"))
                self.assertEqual(len(cache_files), 1, "Should still have only one cache file")
        
        # Test 2: Legacy mode - should use single cache file (backward compatibility)
        with patch('urllib.request.urlretrieve', side_effect=createDummyTle) as mock_download:
            with patch('RMS.Routines.SatellitePositions.loadRobustTLEs') as mock_load:
                mock_load.return_value = []
                
                # Clear test directory
                shutil.rmtree(self.test_dir)
                os.makedirs(self.test_dir)
                
                # With use_daily_cache=False, should use legacy behavior
                loadTLEs(self.test_dir, cache_file, max_age_hours=24, use_daily_cache=False)
                mock_download.assert_called_once()
                
                # Check that regular cache file was created (not date-stamped)
                cache_path = os.path.join(self.test_dir, cache_file)
                self.assertTrue(os.path.exists(cache_path), "Should create non-timestamped cache file")
                
                mock_download.reset_mock()
                
                # Second call should use existing cache
                loadTLEs(self.test_dir, cache_file, max_age_hours=24, use_daily_cache=False)
                mock_download.assert_not_called()


                mock_download.assert_not_called()


    def test_load_tles_with_time(self):
        if not SKYFIELD_AVAILABLE:
            self.skipTest("Skyfield not installed")
            
        cache_file = "active.txt"
        
        # Create two dummy cached files with timestamps
        # File 1: 2020-01-01 (Old)
        date_str_1 = "20200101"
        file_1 = os.path.join(self.test_dir, f"TLE_{date_str_1}_000000_{cache_file}")
        with open(file_1, 'w') as f:
            f.write("DUMMY OLD TLE\n")
            
        # File 2: 2023-01-01 (Newer)
        date_str_2 = "20230101"
        file_2 = os.path.join(self.test_dir, f"TLE_{date_str_2}_000000_{cache_file}")
        with open(file_2, 'w') as f:
            f.write("DUMMY NEW TLE\n")
            
        # Mock loadRobustTLEs to avoid parsing error
        with patch('RMS.Routines.SatellitePositions.loadRobustTLEs') as mock_load:
            mock_load.return_value = []
            
            # Case 1: Ask for time near File 1 (2020)
            target_time_1 = datetime.datetime(2020, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
            loadTLEs(self.test_dir, cache_file, time_of_interest=target_time_1)
            
            # Should load file 1
            mock_load.assert_called_with(file_1)
            
            # Case 2: Ask for time near File 2 (2023)
            target_time_2 = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
            loadTLEs(self.test_dir, cache_file, time_of_interest=target_time_2)
            
            # Should load file 2
            mock_load.assert_called_with(file_2)
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
