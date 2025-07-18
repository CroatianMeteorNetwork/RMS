from __future__ import print_function, division, absolute_import


import os
import sys
import site
import errno
import logging
import logging.handlers
import multiprocessing
import datetime
import threading
import atexit
import time


try:
    from logging.handlers import QueueHandler  # Python 3.2+
except ImportError:
    class QueueHandler(logging.Handler):
        """Minimal backport of logging.handlers.QueueHandler for Python 2."""
        def __init__(self, queue):
            logging.Handler.__init__(self)
            self.queue = queue

        def emit(self, record):
            try:
                self.queue.put_nowait(record)
            except Exception:
                self.handleError(record)

    # Inject into logging.handlers for consistent reference later
    logging.handlers.QueueHandler = QueueHandler


try:
    from logging.handlers import QueueListener  # Python 3.2+
except ImportError:
    class QueueListener(object):
        def __init__(self, queue, *handlers):
            self.queue = queue
            self.handlers = handlers
            self._stop = False

        def start(self):
            # No-op: we pull from the queue manually in _listener_process
            pass

        def stop(self):
            self._stop = True

        def handle(self, record):
            for handler in self.handlers:
                if record is not None:
                    handler.handle(record)

    # Inject into logging.handlers for consistency
    logging.handlers.QueueListener = QueueListener

# Import for getRmsRootDir() function.
if sys.version_info[0] < 3:
    import pkgutil
else:
    import importlib.util


##############################################################################
# GLOBALS
##############################################################################

# Set GStreamer debug level. Use '2' for warnings in production environments.
# Level 4 and above are overwhelming the log
# If higher verbosity is needed, disable in client scripts
if not os.getenv('GST_DEBUG', default=None):
    os.environ['GST_DEBUG'] = '2'

# Add a default stderr handler for pre-initialization log messages
_default_handler = logging.StreamHandler(sys.stderr)
_default_formatter = logging.Formatter('%(message)s')
_default_handler.setFormatter(_default_formatter)
_pre_init_logger = logging.getLogger()
_pre_init_logger.addHandler(_default_handler)
_pre_init_logger.setLevel(logging.INFO)

##############################################################################
# HELPERS
##############################################################################
class LevelRespectingQueueListener(logging.handlers.QueueListener):
    """QueueListener that respects individual handler levels."""
    def handle(self, record):
        """Override to check handler levels before handling."""
        record = self.prepare(record)
        for handler in self.handlers:
            if record.levelno >= handler.level:  # Check handler level
                handler.handle(record)


class LoggerWriter:
    """ Used to redirect stdout/stderr to the log.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


def _inside(path, root):
    """
    Return True if *path* lies inside *root* (string-prefix test).

    We normalize *root* to end with the platform separator so that
    "/opt/RMS_data" does **not** count as inside "/opt/RMS".
    Works on both Python 2.7 and 3.x.
    """
    root = root.rstrip(os.sep) + os.sep   # ensure ".../RMS/" not ".../RMS"
    return os.path.commonprefix([path, root]) == root


class InRmsFilter(logging.Filter):
    """
    Logging filter that keeps only records whose source file lives
    inside the RMS repository tree *and* outside any site-packages
    directory.

    - Records from third-party or standard-library modules (which
      reside under site-packages) are discarded.
    - Records from RMS codebase are allowed through.
    """
    def __init__(self, config):
        super(InRmsFilter, self).__init__()
        self.config = config

        # Initialize allowed directories
        self.allowed_dirs, self.site_dirs = getWhiteAndBlackLists(self.config)

    def filter(self, record):
        p = os.path.realpath(record.pathname)

        # reject std-lib / third-party
        if any(_inside(p, sd) for sd in self.site_dirs):
            return False

        # accept RMS tree **or** external scripts directory
        return any(_inside(p, root) for root in self.allowed_dirs)


# Reproduced from RMS.Misc due to circular import issue
def getRmsRootDir():
    """
        Return the path to the RMS root directory without importing the whole
        codebase
    """
    if sys.version_info[0] == 3:
        # Python 3.x: Use importlib to find the RMS module
        rms_spec = importlib.util.find_spec('RMS')
        if rms_spec is None or rms_spec.origin is None:
            raise ImportError("RMS module not found.")

        # Get the absolute path to the RMS root directory
        return os.path.abspath(os.path.dirname(os.path.dirname(rms_spec.origin)))
    else:
        # Python 2.7: Use pkgutil (deprecated) to locate the RMS module
        loader = pkgutil.get_loader('RMS')
        if loader is None:
            raise ImportError("RMS module not found.")

        # Get the filename associated with the loader
        rms_file = loader.get_filename()

        # Get the absolute path to the RMS root directory
        return os.path.abspath(os.path.dirname(os.path.dirname(rms_file)))


# Reproduced from RMS.Misc due to circular import issue
def mkdirP(path):
    """ Makes a directory and handles all errors.
    
    Arguments:
        path: [str] Directory path to create
        
    Return:
        [bool] True if successful, False otherwise
    """
    try:
        os.makedirs(path)
        return True
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return True
        else:
            print("Error creating directory: " + str(exc))
            return False
    except Exception as e:
        print("Error creating directory: " + str(e))
        return False


# Reproduced from RMS.Misc due to circular import issue
class RmsDateTime:
    """ Use Python-version-specific UTC retrieval.
    """
    if sys.version_info[0] < 3:
        @staticmethod
        def utcnow():
            return datetime.datetime.utcnow()
    else:
        @staticmethod
        def utcnow():
            return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


# Reproduced from RMS.Misc due to circular import issue

class UTCFromTimestamp:
    """Cross-version helper to convert Unix timestamps to naive UTC datetime objects.

    - Python 2.7-3.11: uses datetime.utcfromtimestamp()
    - Python 3.12+: uses datetime.fromtimestamp(..., tz=timezone.utc).replace(tzinfo=None)
    """

    @staticmethod
    def utcfromtimestamp(timestamp):
        if sys.version_info >= (3, 12):
            # Use aware datetime then strip tzinfo to make it naive
            return datetime.datetime.fromtimestamp(
                timestamp, tz=UTCFromTimestamp._get_utc_timezone()
            ).replace(tzinfo=None)
        else:
            return datetime.datetime.utcfromtimestamp(timestamp)

    @staticmethod
    def _get_utc_timezone():
        """Safely provide UTC tzinfo across Python versions."""
        try:
            # Python 3.2+
            from datetime import timezone
            return timezone.utc
        except ImportError:
            # Python 2: no timezone support
            raise NotImplementedError(
                "timezone-aware fromtimestamp() is not supported in Python < 3.2. "
                "Use Python >= 3.12 or fallback to utcfromtimestamp()."
            )



def gstDebugLogger(category, level, file, function, line, obj, message, user_data):
    """ Maps GStreamer debug levels to Python logging levels and logs
        the message directly through the logging system.
    """
    # Get the main logger instance
    logger = logging.getLogger("Logger") 
    
    # Extract message information safely
    cat_name = category.get_name() if category else "Unknown"
    msg_str = message.get() if message else "No message"
    
    # Format and log the message
    log_msg = "{} {}:{:d}:{}: {}".format(cat_name, file, line, function, msg_str)
    logger.info(log_msg)
    return True


##############################################################################
# CUSTOM HANDLER
##############################################################################

class CustomHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Custom handler for rotating log files where the new file's name
    reflects the start time of the new logging period, without renaming old files.
    
    On rollover, it closes the current log file and creates a new one with a
    filename timestamped to the beginning of the new logging interval.
    """
    def __init__(self, station_id, log_file_prefix, filename, when='H', interval=24, utc=True, **kwargs):
        """
        Initializes the handler.
        
        Args:
            station_id (str): The station ID to embed in the filename.
            log_file_prefix (str): A prefix for the log filename.
            filename (str): The initial full path to the log file.
            when, interval, utc, **kwargs: Standard TimedRotatingFileHandler arguments.
        """
        self.station_id = station_id
        self.log_file_prefix = log_file_prefix

        # The namer/rotator attributes are not used since we override doRollover.
        super(CustomHandler, self).__init__(
            filename=filename, when=when, interval=interval, utc=utc, **kwargs
        )

    def doRollover(self):
        """
        Handles rollover by closing the current file and opening a new one with an
        updated timestamp in its name.
        """
        # Close the current file stream
        if self.stream:
            self.stream.close()
            self.stream = None

        # The 'rolloverAt' time is the scheduled time for the rollover, which is
        # the exact start time for the new log file.
        rollover_time_s = self.rolloverAt
        
        if self.utc:
            time_tuple = time.gmtime(rollover_time_s)
        else:
            time_tuple = time.localtime(rollover_time_s)
        
        # Format the time for the new filename
        new_time_str = time.strftime("%Y%m%d_%H%M%S", time_tuple)

        # Construct the new base filename in the same directory.
        # This is the crucial step that changes the name of the NEXT log file.
        self.baseFilename = os.path.join(
            os.path.dirname(self.baseFilename),
            "{}log_{}_{}.log".format(self.log_file_prefix, self.station_id, new_time_str)
        )
        
        # Open the new log file stream using the updated baseFilename.
        self.stream = self._open()

        # Calculate the time for the next rollover.
        # This logic is adapted from the standard library to ensure correctness.
        currentTime = int(time.time())
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval

        # Handle potential Daylight Saving Time shifts for certain rollover schedules
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstNow = time.localtime(currentTime)[-1]
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:
                    addend = -3600
                else:
                    addend = 3600
                newRolloverAt += addend
        
        self.rolloverAt = newRolloverAt



class LoggingManager:
    """Manages the lifecycle of the multiprocessing logger."""
    def __init__(self):
        self.logging_queue = None
        self.listener_process = None
        self.is_initialized = False
        self.init_lock = threading.Lock()

    def initLogging(self, config, log_file_prefix="", safedir=None, 
                    console_level=logging.INFO, file_level=logging.DEBUG):
        """ 
        Spawns the listener process and configures logging. This method is designed to be called once in 
        the main process.
        
        Arguments:
            config: [Config] Configuration object.

        Keyword Arguments:
            log_file_prefix: [str] Prefix for the log file name.
            safedir: [str] Safe directory to use if the log directory is not writable.
            console_level: [int] Logging level for the console output. E.g. logging.INFO
            file_level: [int] Logging level for the file output. E.g. logging.DEBUG

        """

        with self.init_lock:
            if self.is_initialized:
                return

            # Remove any default handlers from the root logger
            main_logger = logging.getLogger()
            for handler in main_logger.handlers[:]:
                main_logger.removeHandler(handler)

            # Spawn the listener process
            self.logging_queue = multiprocessing.Queue(-1)
            self.listener_process = multiprocessing.Process(
                target=_listener_process,
                args=(self.logging_queue, config, log_file_prefix, safedir, console_level, file_level)
            )
            self.listener_process.daemon = True
            self.listener_process.start()
            
            # Configure the queue handler in the main process
            qh = logging.handlers.QueueHandler(self.logging_queue)
            qh.setFormatter(logging.Formatter('%(message)s'))
            qh.addFilter(InRmsFilter(config))

            # Replace root handlers with our queue handler
            main_logger.handlers = [qh]
            main_logger.setLevel(min(console_level, file_level)) # Keep root logger permissive
            main_logger.propagate = False

            # Redirect standard streams
            sys.stderr = LoggerWriter(main_logger, logging.WARNING)
            if config.log_stdout:
                sys.stdout = LoggerWriter(main_logger, logging.INFO)

            self.is_initialized = True
            main_logger.debug("initLogging completed; queue listener started.")

            # Register the instance's shutdown method for clean exit
            atexit.register(self.shutdownLogging)

    def shutdownLogging(self):
        """
        Handles cleanup of logging resources. Stops the listener process.
        """
        with self.init_lock:
            if not self.is_initialized:
                return
            
            # Stop the listener process
            if self.listener_process and self.listener_process.is_alive():
                print("Shutting down logging...") # Added for visibility during shutdown
                self.logging_queue.put(None)  # Sentinel to stop the listener loop
                self.listener_process.join(timeout=5)
                if self.listener_process.is_alive():
                    self.listener_process.terminate()
            
            self.is_initialized = False
            print("Logging shutdown complete.")


##############################################################################
# LISTENER SIDE
##############################################################################

def _listener_configurer(config, log_file_prefix, safedir, console_level=logging.INFO, file_level=logging.DEBUG):
    """ Set up the root logger with a TimedRotatingFileHandler. 
    This runs in the separate listener process.
    """
    # Set DEBUG on root logger - this is the master filter for all handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Set up log directory
    log_path = os.path.join(config.data_dir, config.log_dir)

    # Make directories
    print("Creating directory: " + config.data_dir)
    data_dir_status = mkdirP(config.data_dir)
    print("   Success: {}".format(data_dir_status))
    print("Creating directory: " + log_path)
    log_path_status = mkdirP(log_path)
    print("   Success: {}".format(log_path_status))

    # If the log directory doesn't exist or is not writable, use the safe directory
    if safedir is not None:

        # Make sure the safedir is a directory and not a file
        if os.path.isfile(safedir):
            safedir = os.path.dirname(safedir)

        # Make sure the safe directory exists and is writable
        if not os.path.exists(log_path) or not os.access(log_path, os.W_OK):
            root_logger.debug("Log directory not writable, using safedir: %s", safedir)
            log_path = safedir
            mkdirP(log_path)

    # Generate log filename with timestamp
    start_time_str = RmsDateTime.utcnow().strftime("%Y%m%d_%H%M%S")
    logfile_name = "{}log_{}_{}.log".format(log_file_prefix, config.stationID, start_time_str)
    full_path = os.path.join(log_path, logfile_name)

    # If RMS is to reboot daily, set the rollover time to 25 hours to prevent log fracturing before a new
    # capture session starts
    if config.reboot_after_processing:
        rollover_interval = 30
    else:
        rollover_interval = 24

    # Initialize file and console handlers
    handler = CustomHandler(
        station_id=config.stationID,
        log_file_prefix=log_file_prefix,
        filename=full_path,
        when='H',
        interval=rollover_interval,
        utc=True
    )
    console = logging.StreamHandler(sys.stdout)

    # Set different levels for each handler
    handler.setLevel(file_level)
    console.setLevel(console_level)
    
    # Add filters to both handlers
    handler.addFilter(InRmsFilter(config))
    console.addFilter(InRmsFilter(config))

    # Set common formatter for both handlers
    formatter = logging.Formatter(
        fmt='%(asctime)s-%(levelname)s-%(module)s-line:%(lineno)d - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    console.setFormatter(formatter)

    # Configure root logger with both handlers
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.addHandler(console)
    root_logger.propagate = False
    root_logger.debug("Log listener configured. Current file: %s", full_path)


def _listener_process(queue, config, log_file_prefix, safedir, console_level=logging.INFO, file_level=logging.DEBUG):
    """ Target function for the logging listener process.
    Ignores SIGINT and runs QueueListener for async logging.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Configure the listener process
    logging.Formatter.converter = time.gmtime
    _listener_configurer(config, log_file_prefix, safedir, console_level, file_level)

    # Start queue listener with our custom listener
    main_logger = logging.getLogger()
    queue_listener = LevelRespectingQueueListener(queue, *main_logger.handlers)
    queue_listener.start()

    # Keep the process alive
    while True:
        try:
            record = queue.get()
            if record is None:  # Shutdown sentinel
                break
            queue_listener.handle(record)
        except Exception as e:
            print("Error in listener process: {}".format(e))
            continue

    queue_listener.stop()


##############################################################################
# PUBLIC ENTRY POINT
##############################################################################

def getWhiteAndBlackLists(config):
    """ Returns the whitelisted RMS root and external script directories,
        and the blacklisted site-packages directories.
        
    Return:
        (set, set) Tuple of whitelisted and blacklisted directories
    """

    # Whitelist RMS root and external script directories
    rms_root = os.path.realpath(getRmsRootDir())
    allowed_dirs = {rms_root}
    ext = config.external_script_path
    if ext:
        ext_root = os.path.realpath(ext)
        if not os.path.isdir(ext_root):          # it's a .py file
            ext_root = os.path.dirname(ext_root)
        allowed_dirs.add(ext_root)

    # Blacklist site-packages directories (with Py2 fallback)
    try:
        site_packages = site.getsitepackages()
        user_site    = site.getusersitepackages()
    except (AttributeError, IOError):
        from distutils.sysconfig import get_python_lib
        site_packages = [get_python_lib()]
        user_site     = getattr(site, 'USER_SITE',
                                 get_python_lib(prefix=sys.prefix))

    site_dirs = set(os.path.realpath(p) for p in site_packages)
    site_dirs.add(os.path.realpath(user_site))

    return allowed_dirs, site_dirs


def getLogger(name=None, level="DEBUG", stdout=False):
    """ Get a logger instance.
    
    Arguments:
        name: [str] Logger name. If None, returns "logger"
        level: [str] Logging level to set ("DEBUG","INFO","WARNING","ERROR","CRITICAL")
        stdout: [bool] If True, adds a StreamHandler to stdout
        
    Return:
        [Logger] Logger instance
    """
    logger = logging.getLogger(name if name else "logger")

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(level_map[level.upper()])

    # Add stdout handler if requested
    if stdout:
        out_hdlr = logging.StreamHandler(sys.stdout)
        logger.addHandler(out_hdlr)

    return logger
