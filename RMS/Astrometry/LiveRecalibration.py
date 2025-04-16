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
    
    def __init__(self, config, platepar):
        
        threading.Thread.__init__(self)

        self.config = config
        self.orig_platepar = platepar

        # Thread lifecycle control event
        self._stop_event = threading.Event()
        self.started = False

        # Queue of CALSTARS data to be processed
        self.calstars_queue = queue.Queue(maxsize=1000)

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


    def run(self):
        """
        This method runs in a separate thread and continuously processes star observations
        from the queue, recalibrating the platepars in real-time.
        """

        while not self._stop_event.is_set():
            
            try:
                # Block until there is data in the queue or timeout occurs
                ff_name, star_data = self.calstars_queue.get(timeout=1)

                # Fetch the platepar closest in time to the current FF file
                ff_dt = filenameToDatetime(ff_name)
                working_platepar = self.getClosestPlatepar(ff_dt)

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
                )

                # Safely update the shared platepar dictionary
                with self.recalibrated_platepars_lock:
                    self.recalibrated_platepars.update(recalibrated_platepars)

                print(f"Recalibrated platepar for {ff_name}")
                print("Success:", recalibrated_platepars[ff_name].auto_recalibrated)

            except queue.Empty:
                # No data to process; continue waiting
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


    def addCalstars(self, ff_name, star_data):
        """
        Add CALSTARS data for recalibration to the queue. If the queue is full,
        it will wait until space becomes available.

        Arguments:
            ff_name: [str] Name of the FF file.
            star_data: [list] List of star data to be added for recalibration.
        """

        if not self._stop_event.is_set():
            try:
                self.calstars_queue.put((ff_name, star_data), timeout=1)
                
                return True  # Successfully added to the queue
            

            except queue.Full:
                return False
            
            except Exception as e:
                log.exception(f"Failed to add CALSTARS data for {ff_name}: {e}")
                return False
            
        else:
            log.warning("Recalibration thread is stopped. Cannot add CALSTARS data.")
            return False


    def getClosestPlatepar(self, dt):
        """
        Get the closest platepar based on the given datetime.

        Arguments:
            dt: [datetime] Time to find the closest platepar for.

        Returns:
            platepar: [Platepar] Platepar object closest in time to the given time.
        """

        with self.recalibrated_platepars_lock:

            # If no recalibrated platepars are available, return the original
            if not self.recalibrated_platepars:
                return self.orig_platepar

            # Find the closest recalibrated platepar to the given time
            closest_platepar = None
            closest_time_diff = float('inf')
            for ff_name, platepar in self.recalibrated_platepars.items():
                time_diff = abs((dt - filenameToDatetime(ff_name)).total_seconds())
                if time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    closest_platepar = platepar

        return closest_platepar


    def mapCoordinates(self, coordinates):
        """
        Map coordinates using the most appropriate recalibrated pointing.

        Arguments:
            coordinates: [list] List of coordinates to be mapped.

        Return:
            mapped_coordinates: [list] List of mapped coordinates.
        """
        return coordinates  # Placeholder for actual mapping logic



if __name__ == "__main__":

    import argparse
    import random
    import logging
    import signal

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
    ftp_data = readFTPdetectinfo(*os.path.split(ftpdetectinfo_file))

    # Load the CALSTARS file
    calstars_file = None
    for fn in sorted(os.listdir(cml_args.dir_path)):
        if fn.startswith("CALSTARS"):
            calstars_file = os.path.join(cml_args.dir_path, fn)

    if calstars_file is None:
        raise RuntimeError("No CALSTARS file found in the directory.")
    
    calstars = readCALSTARS(*os.path.split(calstars_file))


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
    try:
        for ff_name, star_data in calstars:
            
            while not recalibrator.addCalstars(ff_name, star_data):
                # If the queue is full, wait and try again
                time.sleep(1)
            
            # time.sleep(random.uniform(2, 5))

    finally:
        recalibrator.stopRecalibration()
