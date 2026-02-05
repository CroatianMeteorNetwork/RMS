
import logging
from logging import config
import multiprocessing as mp
import os
import time
import threading
from RMS.DetectStarsAndMeteors import detectStarsAndMeteorsInVideoFile
from RMS.Astrometry.LiveRecalibration import LiveRecalibration
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
from RMS.Formats.Platepar import findBestPlatepar
from RMS.Misc import mkdirP
from RMS.QueuedPool import QueuedPool


# Get the logger from the main module
log = logging.getLogger("logger")

# Global live recalibrator queue allowing queued pool workers to access the queue 
live_recalibrator_queue = mp.Queue()

def init_manager():
    global live_recalibrator_queue
    if live_recalibrator_queue is None:
        mgr = mp.Manager()
        live_recalibrator_queue = mgr.Queue()
    return live_recalibrator_queue

class RealtimeVideoDetector():
    """ Class to perform real-time video detection on video files as they are created."""


    # A static instance of the detector ued to avoid pickling of the detector in the pooled queue
    _instance = None



    @staticmethod           
    def createDetector(night_data_dir, config, delay_start=0, cores=1, suffix=''):
        """ Static method to create the real-time video detector instance.
        
        Arguments:
            night_data_dir: [str] Path to the directory where night data is stored.
            config: [Config Struct] Cthe current RMS configuration.
            delay_start: [float] Delay before starting the detector (in seconds).
            cores: [int] Number of CPU cores to use. 1 by default.
            suffix: [str] Suffix to append to output files.

        Returns:
            [RealtimeVideoDetector] The created real-time video detector instance.
        """
        if config.realtime_video_detection:
            if config.realtime_video_detection_config is not None:
                config = config.realtime_video_detection_config
                log.info(f'Using a separate configuration for real-time video detection: {config.config_file_name}')
            # Initialize the realtime video detector.  If a configuration override is provided, use it
            RealtimeVideoDetector._instance = RealtimeVideoDetector(night_data_dir, config, delay_start=delay_start, cores=cores, suffix=suffix)
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



    def __init__(self, night_data_dir, config, delay_start=0, cores=1, suffix='realtime'):
        """ Initialize the real-time video detector.  This is a Singleton class

        Arguments:
            night_data_dir: [str] Path to the directory where night data is stored.
            config: [Config Struct]

        Keyword arguments:
            cores: [int] Number of CPU cores to use. None by default. If negative, then the number of cores 
                to be used will be the total number available, minus the given number.
            delay_start: [float] Number of seconds to wait after init before the workers start workings.
            cores: [int] Number of CPU cores to use. 1 by default.
            suffix: [str] Suffix to append to output files. Default: "realtime"

        """
        # The working directory is the configured subdirectory name in the provided night data
        self.night_data_dir = night_data_dir
        self.config = config
        self.realtime_det_dir = os.path.join(night_data_dir, config.realtime_video_detection_night_subdir)
        self.cores = cores
        self.suffix = suffix

        mkdirP(self.realtime_det_dir)
        log.info(f'Realtime video detection directory: {self.realtime_det_dir}')
        self.work_dir = os.path.join(self.realtime_det_dir, 'Work')
        mkdirP(self.work_dir)
        log.info(f'Realtime video detection working directory: {self.work_dir}')


        # Initialize a queue pool to perform the video processing using this classes static processor call
        self.queued_pool = QueuedPool(RealtimeVideoDetector._processvideo, cores=cores, log=log, delay_start=delay_start,
                                       backup_dir=self.work_dir, input_queue_maxsize=None)
  
        # Find the best platepar in the night data directory to use to initialize the live recalibrator
        platepar = findBestPlatepar(self.config, self.night_data_dir)
        # Initialize a live recalibrator for calibrating measurements
        self.live_recalibrator = LiveRecalibration(self.config, platepar, self.realtime_det_dir) 

        # Start the live recalibrator
        log.info('Starting the live recalibrator.')
        self.live_recalibrator.startRecalibration()


    class LiveRecalibratorQueueConsumer(threading.Thread):

        def __init__(self, realtime_detector):
            """ Initialize the live recalibrator queue consumer thread.

            Arguments:
                realtime_detector: [RealtimeVideoDetector] The real-time video detector instance.
            """
            threading.Thread.__init__(self)
            self.realtime_detector = realtime_detector
            self.live_recalibrator = realtime_detector.live_recalibrator
            self.cores = realtime_detector.cores


        def run(self) :
            """ A queue consumer to pass stars and meteors detected in the by queue pool of detector to the live recalibrators Pass detected stars and meteors to the live recalibrator.

            Arguments:
                is_stars: [bool] A flag indicating of the queue entry is a tuple of star data or a detection
                data: [list] List of associated data.
            """
            log.info('Starting live recalibrator queue consumer.')
            while True:
                log.info('Waiting for data from live recalibrator queue...')
                recalibrator_data = live_recalibrator_queue.get()
                # Read the trmination signal
                if recalibrator_data == None:
                    break
                
                # Read the star or detection data from the queue
                is_stars, data = recalibrator_data
                # A helper to try to pass data to live recalibrator, use on an initial write and a retry
                def try_pass_data():    
                    if is_stars:
                        ff_name, star_data, calstars_ff_frames = data
                        added = self.live_recalibrator.addCalstars(ff_name, star_data, calstars_ff_frames)
                        if added:
                            log.info(f'Added CALSTARS data to live recalibrator: {ff_name} with {len(star_data)} measures')
                    else:
                        ff_name, meteor_meas, detection_data = data
                        added = self.live_recalibrator.addMeasurements(ff_name, meteor_meas, detection_data)
                        if added:
                            log.info(f'Added meteor measurements to live recalibrator: {ff_name} with {len(meteor_meas)} measures')
                    return added
                # Repetitvely try to pass data
                added = try_pass_data()
                if not added:
                    log.info('Live recalibrator queue is full, waiting to add data...')
                    wait_time_interval = 0.001
                    wait_time_count = 1
                    time.sleep(wait_time_interval)
                    while not try_pass_data():
                        time.sleep(wait_time_interval)
                        wait_time_count += 1
                    log.info(f'Waited {wait_time_interval * (wait_time_count - 1):.3f} seconds to add data to the live recalibrator queue.')    

            log.info('Live recalibrator queue consumer finished.')



    def start(self):
        """ Start the real-time video detector.
        """
        # Start the recalibrator queue consumer
        init_manager()
        self.live_recalibrator_queue_consumer = RealtimeVideoDetector.LiveRecalibratorQueueConsumer(self)
        self.live_recalibrator_queue_consumer.start()

         # Start the queued pool
        log.info('Starting the real-time video detector queued pool.')
        self.queued_pool.startPool()

    
    def stop(self):
        """ Stop the real-time video detector."""
        
        # Ensure everything is processed and close the queued pool.  This will stop 
        # queing to the live calibrator.
        log.info('Stopping the real-time video detector queued pool.')
        self.queued_pool.closePool()
        log.info('Real-time video detector queued pool stopped.')

        # Stop the live recalibrator queue
        log.info('Stopping live recalibrator queue consumer.')
        # Inject a null entry interpretted by the consumer as a stop request
        live_recalibrator_queue.put(None)
        self.live_recalibrator_queue_consumer.join()
        log.info('Live recalibrator queue consumer stopped.')


        # Wait for the live recalibrator to stop cleanly.
        log.info('Waiting for live recalibrator to finish.')
        self.live_recalibrator.stopRecalibration()
        log.info('Live recalibrator finished.')
        


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
        
        log.info(f'Processing video file for real-time detection: {video_path}')
        init_manager()

        # Perform star and meteor detection constructing Calstars and Detectinfo files in the specified working directory
        output_dir, calstars_name, ftpdetectinfo_name = detectStarsAndMeteorsInVideoFile(video_path, self.config, output_suffix=self.suffix, output_dir=self.work_dir, write_empty=False)

        # Pass the calstars file to the live recalibrator
        if calstars_name is None:
            log.info(f"No CALSTARS data found in {video_path}, skipping live recalibrator update.")
        else:
            calstars, calstars_ff_frames = readCALSTARS(output_dir, calstars_name)
            log.info(f"Extracted CALSTARS data in {calstars_name} with {len(calstars)} stars and {calstars_ff_frames} frames")
            for ff_name, star_data in calstars:
                # Add the CALSTARS data to the queue
                log.info(f"Writing CALSTARS data to live calibrator queue from {calstars_name} for {ff_name} with {len(star_data)} measures")
                live_recalibrator_queue.put( (True, (ff_name, star_data, calstars_ff_frames)) )
    
        # Read the FTPDetectinfo file and pass the measures to the live recalibrator
        if ftpdetectinfo_name is None:
            log.info(f"No FTPdetectinfo data found in {video_path}, skipping live recalibrator update.")
        else:
            det_cam_code,  det_fps, meteor_list = readFTPdetectinfo(output_dir, ftpdetectinfo_name, ret_input_format=True)
            log.info(f"Extracted FTPdetectinfo data in {ftpdetectinfo_name} with {len(meteor_list)} events")
            for meteor_entry in meteor_list:
                ff_name, meteor_No, rho, phi, meteor_meas = meteor_entry
                log.info(f"Writing meteor measurement to live recalibrator queue: {ff_name} meteor {meteor_No} with {len(meteor_meas)} measures")
                # Add the measurements to the recalibrator input queue
                live_recalibrator_queue.put( (False, (ff_name, meteor_meas, [det_cam_code, det_fps, meteor_No, rho, phi])) )
    

if __name__ == "__main__":

    import argparse
    from datetime import timedelta, datetime
    from glob import glob
    import RMS.ConfigReader as cr
    from RMS.Logger import LoggingManager, getLogger

    arg_parser = argparse.ArgumentParser(description='Test capturing frames from a video source defined in the config file. ')
    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, help="Path to a config file which will be used instead of the default one.")
    arg_parser.add_argument('--video_file', metavar='VIDEO_FILE', type=str, help="Path to a video file to be used as a video source")
    arg_parser.add_argument('--video_file_dir', metavar='VIDEO_FILE_DIR', type=str, help="Path to a directory containing video files to be used asvideo sources"
                            " instead of a camera.")
    arg_parser.add_argument('--night_dir', metavar='NIGHT_DIR', type=str, help="Path to a directory where results should be stored.")
    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=1, help="Number of CPU cores to use.")
    arg_parser.add_argument('--prefix', metavar='PREFIX', type=str, default='detection_', help="Prefix to add to log files.")
    arg_parser.add_argument('--suffix', metavar='SUFFIX', type=str, default='', help="Suffix to append to output files.")
    arg_parser.add_argument('--interval', type=float, help="Submission interval in seconds when ingesting multiple videos")
    arg_parser.add_argument('--sync', action='store_true', help="If set, process videos synchronously (one at a time) rather than using multiple cores.")
    arg_parser.add_argument('--skip', action='store_true', help="If set, skip processing of videos and just flag them as processed.")
    arg_parser.add_argument('--continuous_wait_minutes', type=int, help="If specified, continuously monitor the video file directory for new video files to process, waiting this number of minutes between checks.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()
    # A video file source must be provided
    if cml_args.video_file_dir is None and cml_args.video_file is None:
        print ("Either --video_file or --video_file_dir must be provided.")
        exit(1)
    if cml_args.interval is not None and cml_args.interval < 0:
        print ("--interval must be non-negative.")
        exit(1)
    if cml_args.video_file_dir is not None:
        if not os.path.exists(cml_args.video_file_dir):
            print(f"Video file directory does not exist: {cml_args.video_file_dir}")
            exit(1)
        if cml_args.sync and cml_args.skip:
            print("--sync and --skip cannot both be set when processing a video file directory.")
            exit(1)
    else:
        if cml_args.sync or cml_args.skip or cml_args.continuous_wait_minutes is not None:
            print("--sync, --skip, and --continuous can only be set when processing a video file directory.")
            exit(1)

    # Mainline procssing of files, either called once or in a Ctrl-C cancelled continuous loop
    def ProcessFiles():

        # Load the config file
        config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

        # Initialize the logger
        log_manager = LoggingManager()
        log_manager.initLogging(config, cml_args.prefix)

        # Get the logger handle
        log = getLogger("logger")

        # Allocate and start the detector 
        rtvd = RealtimeVideoDetector.createDetector(cml_args.night_dir, config, delay_start=0, cores=cml_args.cores, suffix=cml_args.suffix, synchronous=cml_args.sync)
        rtvd.start()

        # Setup any required video submissioninterval tracking
        interval_delta = None
        if cml_args.interval is not None :
            interval_delta = timedelta(seconds=cml_args.interval)
            last_time = datetime.now() - interval_delta

        # Helper to sumbit a video for processing, either queued or not
        def addVideo(video_file):
            if cml_args.skip:
                return
            elif cml_args.sync:
                rtvd.processVideo(video_file)
            else:
                rtvd.addVideoFile(video_file)
        
        # Helper to provide a video to the realtime detector, either trivially or a time interval
        def addVideoWithInterval(video_file):
            global last_time
            if interval_delta is not None:
                sleep_time = (((last_time + interval_delta) - datetime.now()).total_seconds())
                if sleep_time > 0:
                    log.info(f'Waiting {sleep_time:.3f} seconds before submitting next video file: {video_file}')
                    time.sleep(sleep_time)
                last_time = datetime.now()
            addVideo(video_file)   

        # A helper class to manage what videos have been processed, currently implemented as a simple
        # last video file processes, where it is presumed the videos are processed in file name order.
        # The last processed file name is written to a tracker file in the base directory of where viseos are foun.
        class ProcessedVideoTracker():

            tracker_file_name = os.path.join(cml_args.night_dir, 'last_processed_video.txt')

            def __init__(self, directory):
                self.directory = directory  
                # Initialize the last processed video file name from the tracker file
                if os.path.exists(self.tracker_file_name):
                    with open(self.tracker_file_name, 'r') as tf:
                        self.last_video_file = tf.read().strip()
                else:   
                    self.last_video_file = None

            def isProcessed(self, video_file):
                if self.last_video_file is None:
                    return False
                return (os.path.basename(video_file) <= self.last_video_file)

            def markProcessed(self, video_file):
                filename = os.path.basename(video_file)
                self.last_video_file = filename
                with open(self.tracker_file_name, 'w') as tf:
                    tf.write(filename + '\n')

        # Provide the requested file
        if cml_args.video_file is not None:
            addVideoWithInterval(cml_args.video_file)

        # Process the requested directory
        if cml_args.video_file_dir is not None:
            # Initialize a new file tracker
            processed_tracker = ProcessedVideoTracker(cml_args.video_file_dir)
            # Filter files to ones not yet processed
            files = sorted(glob(os.path.join(cml_args.video_file_dir, "**","*_video.mkv"), recursive=True))
            files = [file for file in files if not processed_tracker.isProcessed(file)]
            # Process each file 
            for file in files:
                addVideoWithInterval(file)
                processed_tracker.markProcessed(file)

        # Stop the detector, it will cleanup processing before exiting  
        rtvd.stop()    

    # The true mainline, either process the files once, of in a cntl-C cancelled loop
    if cml_args.continuous_wait_minutes is None:
        ProcessFiles()
    else:
        log.info('Starting continuous real-time video detection processing loop.  Press Ctrl-C to exit.')
        try:
            while True:
                ProcessFiles()
                log.info(f'Waiting {cml_args.continuous_wait_minutes} minutes before checking for new video files to process.')
                time.sleep(cml_args.continuous_wait_minutes * 60)
        except KeyboardInterrupt:
            log.info('Exiting continuous real-time video detection processing loop on user request.')

