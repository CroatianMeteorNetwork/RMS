
import logging
from logging import config
import os
import time
from RMS.DetectStarsAndMeteors import detectStarsAndMeteorsInVideoFile
from RMS.Astrometry.LiveRecalibration import LiveRecalibration
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
from RMS.Formats.Platepar import findBestPlatepar
from RMS.Misc import mkdirP
from RMS.QueuedPool import QueuedPool


# Get the logger from the main module
log = logging.getLogger("logger")


class RealtimeVideoDetector():
    """ Class to perform real-time video detection on video files as they are created."""


    # A static instance of the detector ued to avoid pickling of the detector in the pooled queue
    _instance = None



    @staticmethod           
    def createDetector(night_data_dir, config, delay_start=0):
        """ Static method to create the real-time video detector instance.
        
        Arguments:
            night_data_dir: [str] Path to the directory where night data is stored.
            config: [Config Struct] Cthe current RMS configuration.
            delay_start: [float] Delay before starting the detector (in seconds).

        Returns:
            [RealtimeVideoDetector] The created real-time video detector instance.
        """
        if config.realtime_video_detection:
            if config.realtime_video_detection_config is not None:
                config = config.realtime_video_detection_config
                log.info(f'Using a separate configuration for real-time video detection: {config.config_file_name}')
            # Initialize the realtime video detector.  If a configuration override is provided, use it
            RealtimeVideoDetector._instance = RealtimeVideoDetector(night_data_dir, config, delay_start=delay_start)
            return RealtimeVideoDetector._instance
        else:
            return None
        
    @staticmethod
    def _processvideo(video_path):
        """ Static method to process a video file using the singlon instance of the real-time video detector.
        
        Arguments:
            video_path: [str] Path to the video file to be processed.
        """
        return RealtimeVideoDetector._instance.processVideo(video_path)



    def __init__(self, night_data_dir, config, delay_start=0):
        """ Initialize the real-time video detector.  This is a Singlon class
        
        Arguments:
            night_data_dir: [str] Path to the directory where night data is stored.
            config: [Config Struct]

        Keyword arguments:
            cores: [int] Number of CPU cores to use. None by default. If negative, then the number of cores 
                to be used will be the total number available, minus the given number.
            delay_start: [float] Number of seconds to wait after init before the workers start workings.

        """
        # The working directory is the configured subdirectory name in the provided night data
        self.night_data_dir = night_data_dir
        self.config = config
        self.realtime_det_dir = os.path.join(night_data_dir, config.realtime_video_detection_night_subdir)

        mkdirP(self.realtime_det_dir)
        log.info(f'Realtime video detection directory: {self.realtime_det_dir}')
        self.work_dir = os.path.join(self.realtime_det_dir, 'Work')
        mkdirP(self.work_dir)
        log.info(f'Realtime video detection working directory: {self.work_dir}')

        # Initialize a queue pool to perform the video processing using this classes static processor call
        self.queued_pool = QueuedPool(RealtimeVideoDetector._processvideo, cores=1, log=log, delay_start=delay_start,
                                       backup_dir=self.work_dir, input_queue_maxsize=None)
  
        # A live recalibrator is allocated in the one and ony process processing the queue pool
        self.live_recalibrator = None


    def start(self):
        """ Start the real-time video detector.
        """
         # Start the queued pool
        log.info('Starting the real-time video detector queue.')
        self.queued_pool.startPool()

    
    def stop(self):
        """ Stop the real-time video detector.
        """
        
        # Ensure everything is processed and close the queued pool.  This will stop 
        # queing to the live calibrator.
        log.info('Stopping the real-time video detector.')
        self.queued_pool.closePool()
        log.info('Real-time video detector stopped.')

        # Wait for the live recalibrator to stop cleanly.
        log.info('Waiting for live recalibrator to finish.')
        self.live_recalibrator.stopRecalibration()
        log.info('Live recalibrator finished.')
        
        # Stop the thread writing recalibrated results to disk
        log.info('Waiting for write thread to finish.')



    def getInputQueueSize(self):
        """ Get the number of video files currently in the input queue.
        Returns:
            [int] The number of video files in the input queue.
        """
        return self.queued_pool.input_queue.qsize()
    

    def deleteBackupFiles(self):
        """ Delete any backup files in the queued pool
        """
        self.queued_pool.deleteBackupFiles()


    def addVideoFile(self, video_path):
        """ Add a video file to the real-time video detector processing pool resulting in an evntual call
        to the static _processVideo() and through the singlon instance processVideo() method.
        Arguments:
            video_path: [str] path to a video file.
        """
        log.info(f'Adding video file for real-time detection: {video_path}')
        self.queued_pool.addJob([video_path])


    def processVideo(self, video_path):
        """ Process a video file for real-time video detection, creating a calstars file and an FTPDetectinfo file
        in the working directory.
        Arguments:
            video_path: [str] path to a video file.
        Return:
            [output_dir, calstars_name, ftpdetectinfo_name]: The output directory and file names
       """
        
        # Allocate and initialize a live recalibrartor in the process processing the queue
        if self.live_recalibrator is None:

            # Find the best platepar in the night data directory to use to initialize the live recalibrator
            platepar = findBestPlatepar(self.config, self.night_data_dir)

            # Initialize a live recalibrator for calibrating measurements
            self.live_recalibrator = LiveRecalibration(self.config, platepar, self.realtime_det_dir) 

            # Start the live recalibrator
            log.info('Starting the live recalibrator.')
            self.live_recalibrator.startRecalibration()



        log.info(f'Processing video file for real-time detection: {video_path}')
        # Perform star and meteor detection constructing Calstars and Detectinfo files in the specified working directory
        output_dir, calstars_name, ftpdetectinfo_name = detectStarsAndMeteorsInVideoFile(video_path, self.config, output_suffix="realtime", output_dir=self.work_dir)

        # Pass the calstars file to the live recalibrator
        calstars, calstars_ff_frames = readCALSTARS(output_dir, calstars_name)
        log.info(f"Passing CALSTARS data to live recalibrator: {calstars_name} with {len(calstars)} stars and {calstars_ff_frames} frames")
        for ff_name, star_data in calstars:
            # Try to add the CALSTARS data to the queue
            log.info(f"Adding CALSTARS data to live recalibrator queue: {ff_name} with {len(star_data)} measures")
            added = self.live_recalibrator.addCalstars(ff_name, star_data, calstars_ff_frames)
            # If the queue is full, wait until space becomes available
            if not added:
                wait_time_interval = 0.001
                wait_time_count = 1
                time.sleep(wait_time_interval)
                # Retry until it succeeds
                while not self.live_recalibrator.addCalstars(ff_name, star_data, calstars_ff_frames):
                    time.sleep(wait_time_interval)
                    wait_time_count += 1
                log.info(f'Waited {wait_time_interval * (wait_time_count - 1):.3f} seconds to add CALSTARS data to the live recalibrator queue.')

        # Read the FTPDetectinfo file and pass the measures to the live recalibrator
        det_cam_code,  det_fps, meteor_list = readFTPdetectinfo(output_dir, ftpdetectinfo_name, ret_input_format=True)
        log.info(f"Passing FTPdetectinfo data to live recalibrator with: {ftpdetectinfo_name} with {len(meteor_list)} events")
        for meteor_entry in meteor_list:
            ff_name, meteor_No, rho, phi, meteor_meas = meteor_entry
            log.info(f"Adding meteor measurement to live recalibrator queue: {ff_name} meteor {meteor_No} with {len(meteor_meas)} measures")
            # Add the measurements to the recalibrator input queue
            added = self.live_recalibrator.addMeasurements(ff_name, meteor_meas, [det_cam_code, det_fps, meteor_No, rho, phi])
            if not added:
                wait_time_interval = 0.001
                wait_time_count = 1
                time.sleep(wait_time_interval)
                while not self.live_recalibrator.addMeasurements(ff_name, meteor_meas, [det_cam_code, det_fps, meteor_No, rho, phi]):
                    time.sleep(wait_time_interval)
                    wait_time_count += 1
                log.info(f'Waited {wait_time_interval * (wait_time_count - 1):.3f} seconds to add FTPdetectinfo data to the live recalibrator queue.')


if __name__ == "__main__":

    import argparse
    from glob import glob
    import RMS.ConfigReader as cr
    from RMS.Logger import LoggingManager, getLogger

    arg_parser = argparse.ArgumentParser(description='Test capturing frames from a video source defined in the config file. ')
    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, help="Path to a config file which will be used instead of the default one.")
    arg_parser.add_argument('--video_file', metavar='VIDEO_FILE', type=str, help="Path to a video file to be used as a video source")
    arg_parser.add_argument('--video_file_dir', metavar='VIDEO_FILE_DIR', type=str, help="Path to a directory containing video files to be used asvideo sources"
                            " instead of a camera.")
    arg_parser.add_argument('--night_dir', metavar='NIGHT_DIR', type=str, help="Path to a directory where results should be stored.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()
    # A video file source must be provided
    if cml_args.video_file_dir is None and cml_args.video_file is None:
        arg_parser.print_help()
        exit(1)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    # Get the logger handle
    log = getLogger("logger")

    rtvd = RealtimeVideoDetector.createDetector(cml_args.night_dir, config, delay_start=0)
    rtvd.start()
    if cml_args.video_file is not None:
        rtvd.processVideo(cml_args.video_file)
    if cml_args.video_file_dir is not None:
        for file in sorted(glob(os.path.join(cml_args.video_file_dir, "**","*_video.mkv"), recursive=True)):
            #tvd.addVideoFile(file)
            rtvd.processVideo(file)
    rtvd.stop()    


