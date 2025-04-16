"""
Class that allows spawning a thread that recives an initial platepar and recalibrates in real time as
star observations come in. In addition, it has a method that allows coordinate mapping using the most
appropriate recalibrated pointing.
"""

import os
import threading
import time
import queue
import logging
import hashlib
import json


from RMS.Astrometry.ApplyAstrometry import applyPlateparToCentroids
from RMS.Astrometry.ApplyRecalibrate import recalibratePlateparsForFF
from RMS.Astrometry.Conversions import date2JD
from RMS.Formats.FFfile import filenameToDatetime, getMiddleTimeFF
from RMS.Formats.StarCatalog import readStarCatalog

# Get the logger from the main module
log = logging.getLogger("logger")
log.setLevel(logging.INFO)


class LiveRecalibration(threading.Thread):
    """
    Class that allows spawning a thread that recives an initial platepar and recalibrates in real time as
    star observations come in. In addition, it has a method that allows coordinate mapping using the most
    appropriate recalibrated pointing.
    """
    
    def __init__(self, config, platepar, meas_triggered_recalib=True, meas_trigger_dt=30):
        """
        Initialize the LiveRecalibration class.

        Arguments:
            config: [Config] Configuration object containing the necessary parameters.
            platepar: [Platepar] Initial platepar to be used for recalibration.

        Keyword Arguments:
            meas_triggered_recalib: [bool] If True, only recalibrate platepars that are close to measurements
                in the measurement queue. If False, all platepars are recalibrated. Default is True.
            meas_trigger_dt: [int] Minimum time difference in seconds to trigger recalibration for 
                measurements. Default is 30 seconds.
        """
        
        threading.Thread.__init__(self)

        self.config = config
        self.orig_platepar = platepar

        self.meas_triggered_recalib = meas_triggered_recalib
        self.meas_trigger_dt = meas_trigger_dt

        # Thread lifecycle control event
        self._stop_event = threading.Event()
        self.started = False

        # Queue of CALSTARS data to be processed (no limit)
        self.calstars_queue = queue.Queue()

        # Queue of measurements to be processed (max size of 10,000)
        self.measurement_input_queue = queue.Queue(maxsize=10_000)

        # Queue of measurements output (max size of 10,000)
        self.measurement_output_queue = queue.Queue(maxsize=10_000)


        # Dictionary of recalibrated platepars (keys are FF names)
        self.recalibrated_platepars = {}
        self.recalibrated_platepars_lock = threading.Lock()


        ### LOAD THE STAR CATALOG ###

        # Load catalog stars (overwrite the mag band ratios if specific catalog is used)
        star_catalog_status = readStarCatalog(
            config.star_catalog_path,
            config.star_catalog_file,
            lim_mag=config.catalog_mag_limit + 1, # Increase the limiting magnitude for the catalog
            mag_band_ratios=config.star_catalog_band_ratios,
        )

        if not star_catalog_status:
            raise RuntimeError("Failed to load star catalog.")

        self.catalog_stars, _, config.star_catalog_band_ratios = star_catalog_status

        ### ###


    # Helper function to generate a unique hash for measurement
    @staticmethod
    def _hash_measurement(ff_name, meteor_meas):
        key_data = {"ff_name": ff_name, "meas": meteor_meas}
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def run(self):
        """
        This method runs in a separate thread and continuously processes star observations
        from the queue, recalibrating the platepars in real-time.
        """

        while not self._stop_event.is_set():


            # Only do the recalibrations of the platepars close to the measurements
            # i.e. recalibrations will be triggered by measurements in the queue
            if self.meas_triggered_recalib:

                # Go through the measurements queue and check if there are any calibration stars close in
                # time to the measurements
                try:
                    # Pull all new measurements and associate them with timestamps and unique hash keys
                    measurements = {}
                    while not self.measurement_input_queue.empty():
                        ff_name, meas = self.measurement_input_queue.get_nowait()
                        meas_time = filenameToDatetime(ff_name)
                        key = self._hash_measurement(ff_name, meas)
                        measurements[key] = (ff_name, meas, meas_time)

                    # If there are no measurements, continue to wait
                    if not measurements:
                        time.sleep(0.1)
                        continue

                    # Drain the CALSTARS queue to get all available calibration data
                    calstars_entries = []
                    while not self.calstars_queue.empty():
                        try:
                            ff_name, star_data, calstars_ff_frames = self.calstars_queue.get_nowait()
                            ff_dt = filenameToDatetime(ff_name)
                            calstars_entries.append((ff_name, ff_dt, star_data, calstars_ff_frames))
                        except queue.Empty:
                            break

                    # Sort the calibration stars by their datetime (newest first)
                    calstars_entries.sort(key=lambda x: x[1], reverse=True)

                    # Match each measurement with the closest calibration star record within the trigger time 
                    # window
                    for key, (m_ff_name, m_meas, m_dt) in measurements.items():
                        closest_entry = None
                        closest_diff = float('inf')

                        # Iterate through the calibration stars to find the closest match
                        for entry in calstars_entries:
                            ff_name, ff_dt, _, _ = entry
                            diff = abs((ff_dt - m_dt).total_seconds())
                            if diff < closest_diff and diff <= self.meas_trigger_dt:
                                closest_diff = diff
                                closest_entry = entry

                            # If the difference is less than 5.12 seconds (128 frames / 25 FPS), break the 
                            # loop
                            if diff <= 5.12:
                                break

                        # If no matching CALSTARS was found, try to use a previously recalibrated platepar
                        if closest_entry is None:

                            with self.recalibrated_platepars_lock:

                                for ff_name, platepar in self.recalibrated_platepars.items():

                                    ff_dt = filenameToDatetime(ff_name)
                                    diff = abs((ff_dt - m_dt).total_seconds())

                                    if diff < closest_diff and diff <= self.meas_trigger_dt:

                                        closest_diff = diff
                                        closest_entry = (ff_name, ff_dt, None, None)  # No need to recalibrate again

                        # If a suitable match was found, recalibrate if not done already
                        if closest_entry is not None:
                            ff_name, ff_dt, star_data, calstars_ff_frames = closest_entry

                            # Check if the recalibration for this file has already been done
                            with self.recalibrated_platepars_lock:
                                already_done = ff_name in self.recalibrated_platepars

                            # Perform recalibration only if this file hasn't been processed yet
                            if not already_done and star_data is not None:

                                try:
                                    # Fetch the platepar closest in time to the current FF file
                                    working_platepar, _ = self.getClosestPlatepar(ff_dt)

                                    # Run the recalibration procedure
                                    recalibrated_platepars = recalibratePlateparsForFF(
                                        working_platepar,
                                        [ff_name],
                                        {ff_name: star_data},
                                        self.catalog_stars,
                                        self.config,
                                        lim_mag=self.config.catalog_mag_limit + 1,
                                        ignore_distance_threshold=False,
                                        ignore_max_stars=False,
                                        ff_frames=calstars_ff_frames,
                                    )

                                    # Safely update the shared platepar dictionary
                                    with self.recalibrated_platepars_lock:
                                        self.recalibrated_platepars.update(recalibrated_platepars)


                                    print(f"Recalibrated platepar for {ff_name}")
                                    print("Success:", recalibrated_platepars[ff_name].auto_recalibrated)


                                except Exception as e:
                                    log.exception(f"Recalibration failed for {ff_name}: {e}")

                            # Perform coordinate mapping using the updated or cached platepar
                            mapped = self.mapCoordinates(m_ff_name, m_meas, 
                                                         pp_time_diff_limit=self.meas_trigger_dt)
                            
                            # If mapping was successful, put the result in the output queue
                            if mapped is not None:
                                self.measurement_output_queue.put((key, m_ff_name, mapped))


                    # Requeue CALSTARS entries that were not used in this pass
                    for entry in calstars_entries:
                        try:
                            self.calstars_queue.put_nowait((entry[0], entry[2], entry[3]))
                        except queue.Full:
                            log.warning(f"CALSTARS requeue full, dropping data for {entry[0]}")

                except Exception as e:
                    log.exception("Error during measurement-triggered recalibration loop.")



            # Process all the calibration stars in the queue, don't wait for measurements
            else:

                try:
                    # Block until there is data in the queue or timeout occurs
                    ff_name, star_data, calstars_ff_frames = self.calstars_queue.get(timeout=1)

                    # Fetch the platepar closest in time to the current FF file
                    ff_dt = filenameToDatetime(ff_name)
                    working_platepar, _ = self.getClosestPlatepar(ff_dt)

                    calstars = {ff_name: star_data}

                    # Perform the recalibration for the given data
                    recalibrated_platepars = recalibratePlateparsForFF(
                        working_platepar,
                        [ff_name],
                        calstars,
                        self.catalog_stars,
                        self.config,
                        lim_mag=self.config.catalog_mag_limit + 1,
                        ignore_distance_threshold=False,
                        ignore_max_stars=False,
                        ff_frames=calstars_ff_frames,
                    )

                    # Safely update the shared platepar dictionary
                    with self.recalibrated_platepars_lock:
                        self.recalibrated_platepars.update(recalibrated_platepars)

                    print(f"Recalibrated platepar for {ff_name}")
                    print("Success:", recalibrated_platepars[ff_name].auto_recalibrated)

                except queue.Empty:
                    # No data to process; continue waiting
                    time.sleep(0.01)
                    continue

                except Exception as e:
                    # Log any unexpected exceptions
                    log.exception(f"Recalibration failed for {ff_name if 'ff_name' in locals() else '[unknown]'}: {e}")


    def startRecalibration(self):
        """
        Start the recalibration thread. Ensures it's only started once.
        """

        if self.started:
            log.warning("Recalibration thread already started.")
            return

        self.started = True
        self.start()


    def stopRecalibration(self):
        """
        Stop the recalibration thread and wait for it to finish cleanly.
        """
        self._stop_event.set()
        self.join()


    def addCalstars(self, ff_name, star_data, calstars_ff_frames):
        """
        Add CALSTARS data for recalibration to the queue. If the queue is full,
        it will wait until space becomes available.

        Arguments:
            ff_name: [str] Name of the FF file.
            star_data: [list] List of star data to be added for recalibration.
            calstars_ff_frames: [int] Number of frames in the FF file or frame chunk on which the stars were 
                detected.
        """

        if not self._stop_event.is_set():
            try:
                self.calstars_queue.put((ff_name, star_data, calstars_ff_frames), timeout=1)
                
                return True  # Successfully added to the queue
            

            except queue.Full:
                return False
            
            except Exception as e:
                log.exception(f"Failed to add CALSTARS data for {ff_name}: {e}")
                return False
            
        else:
            log.warning("Recalibration thread is stopped. Cannot add CALSTARS data.")
            return False
        

    def addMeasurements(self, ff_name, meteor_meas):
        """
        Add meteor measurements for recalibration to the queue. If the queue is full,
        it will wait until space becomes available.

        Arguments:
            ff_name: [str] Name of the FF file.
            meteor_meas: [list] List of meteor measurements to be added for recalibration.
        """

        if not self._stop_event.is_set():
            try:
                self.measurement_input_queue.put((ff_name, meteor_meas), timeout=1)

                return True  # Successfully added to the queue
            

            except queue.Full:
                return False
            
            except Exception as e:
                log.exception(f"Failed to add measurements for {ff_name}: {e}")
                return False
            
        else:
            log.warning("Recalibration thread is stopped. Cannot add measurements.")
            return False


    def getClosestPlatepar(self, dt):
        """
        Get the closest platepar based on the given datetime.

        Arguments:
            dt: [datetime] Time to find the closest platepar for.

        Returns:
            platepar: [Platepar] Platepar object closest in time to the given time.
            time_diff: [float] Time difference in seconds between the given time and the closest platepar.
        """

        with self.recalibrated_platepars_lock:

            # If no recalibrated platepars are available, return the original
            if not self.recalibrated_platepars:
                return self.orig_platepar, None

            # Find the closest recalibrated platepar to the given time
            closest_platepar = None
            closest_time_diff = float('inf')
            for ff_name, platepar in self.recalibrated_platepars.items():
                time_diff = abs((dt - filenameToDatetime(ff_name)).total_seconds())
                if time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    closest_platepar = platepar

        return closest_platepar, closest_time_diff


    def mapCoordinates(self, ff_name, meteor_meas, pp_time_diff_limit=None):
        """
        Map coordinates using the most appropriate recalibrated pointing.

        Arguments:
            ff_name: [str] Name of the FF file.
            meteor_meas: [list] List of meteor measurements to be mapped (from the FTPdetectino file).

        Keyword Arguments:
            time_limit: [datetime] Limit the time difference to the closest platepar. Units in seconds.
                If None, no limit is applied (default).

        Return:
            mapped_coordinates: [list] List of mapped coordinates.
        """

        # Get the closest platepar for the given FF file
        ff_dt = filenameToDatetime(ff_name)
        working_platepar, closest_time = self.getClosestPlatepar(ff_dt)

        if pp_time_diff_limit is not None:

            # If the closest time is -1, it means no recalibrated platepars was found
            if closest_time is None:
                log.warning(f"No recalibrated platepar found for {ff_name}.")
                return None

            # Check if the time difference is within the specified limit
            if abs(closest_time) > pp_time_diff_limit:
                log.warning(f"Time difference ({closest_time} seconds) exceeds limit ({pp_time_diff_limit} seconds).")
                return None

        # Apply the platepar to the meteor measurements
        meteor_picks = applyPlateparToCentroids(
            ff_name, config.fps, meteor_meas, working_platepar, add_calstatus=True
            )
        
        return meteor_picks



if __name__ == "__main__":

    import argparse
    import random
    import logging
    import signal
    import collections

    from RMS.ConfigReader import loadConfigFromDirectory
    from RMS.Formats.Platepar import findBestPlatepar
    from RMS.Formats.CALSTARS import readCALSTARS
    from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile, readFTPdetectinfo
    from RMS.Logger import initLogging

    parser = argparse.ArgumentParser(
        description="Test the LiveRecalibration class using data in the given directory. ")
    
    parser.add_argument("dir_path", type=str, help="Path to the directory containing the data.")

    cml_args = parser.parse_args()


    ### LOAD FILES ###

    # Load configuration and platepar files
    config = loadConfigFromDirectory(".", dir_path=cml_args.dir_path)
    if config is None:
        raise RuntimeError("No config file found in the directory.")
    
    platepar = findBestPlatepar(config, night_data_dir=cml_args.dir_path)
    if platepar is None:
        raise RuntimeError("No platepar file found in the directory.")
    
    # Load FTPdetectinfo file
    ftpdetectinfo_file = findFTPdetectinfoFile(cml_args.dir_path)
    _, _, meteor_list = readFTPdetectinfo(*os.path.split(ftpdetectinfo_file), ret_input_format=True)

    # Load the CALSTARS file
    calstars_file = None
    for fn in sorted(os.listdir(cml_args.dir_path)):
        if fn.startswith("CALSTARS"):
            calstars_file = os.path.join(cml_args.dir_path, fn)

    if calstars_file is None:
        raise RuntimeError("No CALSTARS file found in the directory.")
    
    calstars, calstars_ff_frames = readCALSTARS(*os.path.split(calstars_file))


    # Initialize the logger
    initLogging(config, 'live_recalibrate_', safedir=cml_args.dir_path)
    log = logging.getLogger("logger")
    log.setLevel(logging.INFO)


    ### TEST LIVE RECALIBRATION ###

    # Create and start the recalibration thread
    recalibrator = LiveRecalibration(config, platepar)
    recalibrator.startRecalibration()


    # Define interrupt signal behavior (Ctrl+C)
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        recalibrator.stopRecalibration()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Simulate input of star observations for recalibration
    #for ff_name, star_data in calstars[1100:]:
    for ff_name, star_data in calstars:

        # Try to add the CALSTARS data to the queue
        added = recalibrator.addCalstars(ff_name, star_data, calstars_ff_frames)
        
        # If the queue is full, wait until space becomes available
        if not added:
            
            # Retry until it succeeds
            while not recalibrator.addCalstars(ff_name, star_data, calstars_ff_frames):
                time.sleep(0.001)

    
    # # Map the first few meteor measurements. Wait until the appropriate platepar is available.
    # for meteor_entry in meteor_list[:5]:

    #     ff_name, meteor_No, rho, phi, meteor_meas = meteor_entry

    #     # Apply the recalibrated platepar to meteor centroids
    #     # Keep trying until the recalibrator has a platepar available
    #     status = None
    #     while status is None:
    #         status = recalibrator.mapCoordinates(ff_name, meteor_meas, pp_time_diff_limit=30)

    #         if status is None:
    #             print()
    #             print(f"Waiting for recalibrated platepar for {ff_name}...")
                
    #             # Wait for a short time before trying again
    #             time.sleep(1)

    #     meteor_picks = status
    #     print()
    #     print(f"Mapped coordinates for {ff_name}: {meteor_picks}")


    # Prepare measurements and track them using unique keys
    expected_keys = collections.OrderedDict()

    # Add meteor measurements to the recalibrator input queue
    for meteor_entry in meteor_list:
        ff_name, meteor_No, rho, phi, meteor_meas = meteor_entry

        # Generate a unique key for this measurement
        key = recalibrator._hash_measurement(ff_name, meteor_meas)
        expected_keys[key] = (ff_name, meteor_meas)

        # Add the measurement to the recalibrator input queue
        while not recalibrator.addMeasurements(ff_name, meteor_meas):
            time.sleep(0.001)  # Retry if the queue is temporarily full


    # === Wait for all measurements to be processed ===

    received_results = {}

    while len(received_results) < len(expected_keys):
        try:
            # Get processed result from output queue
            key, ff_name, mapped_coords = recalibrator.measurement_output_queue.get(timeout=5)
            received_results[key] = (ff_name, mapped_coords)
        except queue.Empty:
            print("Still waiting for some mapped measurements...")
            time.sleep(0.5)


    # === Print final results ===

    print("\n=== Mapped Meteor Measurements ===")
    for key, (ff_name, _) in expected_keys.items():
        _, result = received_results[key]
        print(f"{ff_name}: {result}")

    
    recalibrator.stopRecalibration()
