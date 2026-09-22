
import unittest
import os
import shutil
import datetime
import tempfile
import glob
from unittest.mock import patch

from RMS.Routines.SatellitePositions import loadTLEs, SatellitePredictor, SKYFIELD_AVAILABLE
from RMS.Formats.Platepar import Platepar

# A valid TLE (ISS) used as the content of mocked downloads
VALID_TLE = (
    "ISS (ZARYA)\n"
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991\n"
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482\n"
)


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
            """ Side effect for the mock that actually creates the cache file.

            Arguments:
                url: [str] Ignored, the URL the real download would fetch.
                filepath: [str] Path of the cache file to create.
            """

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(VALID_TLE)
        
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
        """ With a time of interest the cached TLE file closest to that time is used instead of
            downloading. """
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


    def test_load_tles_offline_falls_back_to_newest_group_cache(self):
        """ A failed download falls back to the newest cache file of the same TLE group. """
        if not SKYFIELD_AVAILABLE:
            self.skipTest("Skyfield not installed")

        # Old caches of the requested group and a newer one of another group
        file_old = os.path.join(self.test_dir, "TLE_20200101_000000_active.txt")
        file_newer = os.path.join(self.test_dir, "TLE_20200105_000000_active.txt")
        file_other = os.path.join(self.test_dir, "TLE_20200110_000000_starlink.txt")
        for file_path in (file_old, file_newer, file_other):
            with open(file_path, 'w') as f:
                f.write("DUMMY TLE\n")

        # Simulate a download which writes a partial file and then fails
        def failingDownload(url, filepath):
            with open(filepath, 'w') as f:
                f.write("1 25544U 98067A   19343.6")
            raise IOError("connection reset")

        with patch('urllib.request.urlretrieve', side_effect=failingDownload):
            with patch('RMS.Routines.SatellitePositions.loadRobustTLEs') as mock_load:
                mock_load.return_value = []

                loadTLEs(self.test_dir, "active.txt", max_age_hours=24, use_daily_cache=True)

                # The newest cache of the same group must be used
                mock_load.assert_called_once_with(file_newer)

        # No truncated or partial file may be left behind
        remaining = sorted(os.listdir(self.test_dir))
        self.assertEqual(remaining, sorted(os.path.basename(p) for p in (file_old, file_newer, file_other)))


    def test_load_tles_offline_without_cache(self):
        """ A failed download with no cache of the group returns an empty list and leaves no file. """
        if not SKYFIELD_AVAILABLE:
            self.skipTest("Skyfield not installed")

        # Only a cache of another group exists
        with open(os.path.join(self.test_dir, "TLE_20200110_000000_starlink.txt"), 'w') as f:
            f.write("DUMMY TLE\n")

        with patch('urllib.request.urlretrieve', side_effect=IOError("offline")):
            with patch('RMS.Routines.SatellitePositions.loadRobustTLEs') as mock_load:
                self.assertEqual(loadTLEs(self.test_dir, "active.txt", use_daily_cache=True), [])
                mock_load.assert_not_called()

        self.assertEqual(os.listdir(self.test_dir), ["TLE_20200110_000000_starlink.txt"])


    def test_find_closest_tle_file_respects_group(self):
        """ The closest file must be taken from the requested TLE group only. """

        from RMS.Routines.SatellitePositions import findClosestTLEFile

        file_active = os.path.join(self.test_dir, "TLE_20200101_000000_active.txt")
        file_starlink = os.path.join(self.test_dir, "TLE_20200102_000000_starlink.txt")
        for file_path in (file_active, file_starlink):
            with open(file_path, 'w') as f:
                f.write("DUMMY TLE\n")

        target_time = datetime.datetime(2020, 1, 2, 0, 0, 0, tzinfo=datetime.timezone.utc)

        self.assertEqual(findClosestTLEFile(self.test_dir, target_time, cache_file_name="active.txt"),
                         file_active)

        # Without a group name any TLE file is considered (e.g. a user supplied TLE directory)
        self.assertEqual(findClosestTLEFile(self.test_dir, target_time), file_starlink)


    def test_tle_group_matching_is_exact(self):
        """ The files of a group whose name ends with the requested name must not be used. """
        if not SKYFIELD_AVAILABLE:
            self.skipTest("Skyfield not installed")

        from RMS.Routines.SatellitePositions import findClosestTLEFile, findNewestTLECacheFile

        file_active = os.path.join(self.test_dir, "TLE_20200101_000000_active.txt")
        file_gp_active = os.path.join(self.test_dir, "TLE_20200102_000000_gp_active.txt")
        for file_path in (file_active, file_gp_active):
            with open(file_path, 'w') as f:
                f.write(VALID_TLE)

        target_time = datetime.datetime(2020, 1, 2, 0, 0, 0, tzinfo=datetime.timezone.utc)
        self.assertEqual(findClosestTLEFile(self.test_dir, target_time, cache_file_name="active.txt"),
                         file_active)
        self.assertEqual(findNewestTLECacheFile(self.test_dir, "active.txt"), file_active)


    def test_find_closest_tle_file_skips_empty_files(self):
        """ An empty cache file is never the closest file. """

        from RMS.Routines.SatellitePositions import findClosestTLEFile

        file_good = os.path.join(self.test_dir, "TLE_20200101_000000_active.txt")
        file_empty = os.path.join(self.test_dir, "TLE_20200102_000000_active.txt")
        with open(file_good, 'w') as f:
            f.write(VALID_TLE)
        open(file_empty, 'w').close()

        target_time = datetime.datetime(2020, 1, 2, 0, 0, 0, tzinfo=datetime.timezone.utc)
        self.assertEqual(findClosestTLEFile(self.test_dir, target_time, cache_file_name="active.txt"),
                         file_good)


    def test_non_tle_download_is_not_cached(self):
        """ A non-empty response without TLEs (e.g. a rate-limit page) must not become a cache file. """
        if not SKYFIELD_AVAILABLE:
            self.skipTest("Skyfield not installed")

        file_old = os.path.join(self.test_dir, "TLE_20200101_000000_active.txt")
        with open(file_old, 'w') as f:
            f.write(VALID_TLE)

        def rateLimitPage(url, filepath):
            with open(filepath, 'w') as f:
                f.write("GP data has not updated since your last successful download. Try again later.\n")

        with patch('urllib.request.urlretrieve', side_effect=rateLimitPage):
            with patch('RMS.Routines.SatellitePositions.loadRobustTLEs') as mock_load:
                mock_load.return_value = []
                loadTLEs(self.test_dir, "active.txt", use_daily_cache=True)
                mock_load.assert_called_once_with(file_old)

        self.assertEqual(os.listdir(self.test_dir), [os.path.basename(file_old)])


    def test_partial_downloads_are_cleaned(self):
        """ Stale .part files are removed, and an interrupted download leaves none behind. """
        if not SKYFIELD_AVAILABLE:
            self.skipTest("Skyfield not installed")

        # A stale partial download of this group and one of another group
        stale = os.path.join(self.test_dir, "TLE_20200101_000000_active.txt.part")
        other = os.path.join(self.test_dir, "TLE_20200101_000000_starlink.txt.part")
        for file_path in (stale, other):
            with open(file_path, 'w') as f:
                f.write("1 25544U")

        def interruptedDownload(url, filepath):
            with open(filepath, 'w') as f:
                f.write("1 25544U 98067A")
            raise KeyboardInterrupt()

        with patch('urllib.request.urlretrieve', side_effect=interruptedDownload):
            with self.assertRaises(KeyboardInterrupt):
                loadTLEs(self.test_dir, "active.txt", use_daily_cache=True)

        self.assertEqual(os.listdir(self.test_dir), [os.path.basename(other)])
