from __future__ import print_function, division, absolute_import


import os
import sys
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



##############################################################################
# GLOBALS
##############################################################################

# Set GStreamer debug level. Use '2' for warnings in production environments.
# Level 4 and above are overwhelming the log
# If higher verbosity is needed, disable in client scripts
os.environ['GST_DEBUG'] = '2'

_rms_logging_queue = None
_rms_listener_process = None
_rms_init_lock = threading.Lock()
_rms_logger_initialized = False


class PreInitNoiseFilter(logging.Filter):
    """ Filter out noisy messages from specific patterns before logger initialization.
        C/C++ level libraries messages cannot be removed.
        These filters will be removed when proper logging is initialized
    """
    def __init__(self):
        super(PreInitNoiseFilter, self).__init__()

        self.noisy_patterns = [
            "Unable to register",
            "Creating converter",
            "CACHEDIR",
            "No `name` configuration",
            "running build_ext",
            "skipping 'RMS",
            "extension (up-to-date)"
        ]

    def filter(self, record):
        # Check if message contains any noisy pattern
        try:
            message = record.getMessage()
            if any(pattern in message for pattern in self.noisy_patterns):
                return False
        except:
            pass  # If getMessage() fails, continue with module check

        return True


# Add a default stderr handler for pre-initialization log messages
_default_handler = logging.StreamHandler(sys.stderr)
_default_formatter = logging.Formatter('%(message)s')
_default_handler.setFormatter(_default_formatter)
_default_handler.addFilter(PreInitNoiseFilter())
_pre_init_logger = logging.getLogger()
_pre_init_logger.addHandler(_default_handler)
_pre_init_logger.setLevel(logging.DEBUG)

# This handler will be removed when proper logging is initialized


##############################################################################
# HELPERS
##############################################################################

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

    - Python 2.7–3.11: uses datetime.utcfromtimestamp()
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
    """ Custom handler for rotating log files.
    
    The live file: log_XX0000_20241229_112347.log
    On rollover: log_XX0000_20241229_112347-[29_1123-to-30_1123].log
    """
    def __init__(self, station_id, start_time_str, *args, **kwargs):
        self.station_id = station_id
        self.start_time_str = start_time_str
        super(CustomHandler, self).__init__(*args, **kwargs)
        self.suffix = "%Y%m%d_%H%M%S"
        self.namer = self._rename_on_rollover

    def _rename_on_rollover(self, default_name):
        # Parse the default filename
        base_dir, base_file = os.path.split(default_name)
        base_noext, dot, start_time_str = base_file.rpartition('.')
        
        if base_noext.endswith('.log'):
            base_noext = base_noext[:-4]
        
        # Calculate time range for the log file
        start_time = datetime.datetime.strptime(start_time_str, "%Y%m%d_%H%M%S")
        end_time = UTCFromTimestamp.utcfromtimestamp(self.rolloverAt)
        
        # Format the new filename with time range
        start_str = start_time.strftime("%d_%H%M")
        end_str = end_time.strftime("%d_%H%M")
        new_name = "{}-[{}-to-{}].log".format(base_noext, start_str, end_str)
        
        return os.path.join(base_dir, new_name)


##############################################################################
# LISTENER SIDE
##############################################################################

class NoiseFilter(logging.Filter):
    """ Filter out noisy messages from specific modules.
    """
    def __init__(self):
        super(NoiseFilter, self).__init__()
        self.noisy_modules = {'font_manager', 'ticker', 'transport', 'sftp', 'dvrip', 'channel', 'cmd', 'PngImagePlugin'}

    def filter(self, record):
        if record.levelno in (logging.DEBUG, logging.INFO) and record.module in self.noisy_modules:
            return False
        return True


def _listener_configurer(config, log_file_prefix, safedir):
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

    # Initialize file and console handlers
    handler = CustomHandler(
        station_id=config.stationID,
        start_time_str=start_time_str,
        filename=full_path,
        when='H',
        interval=24,
        utc=True
    )
    console = logging.StreamHandler(sys.stdout)

    # Add noise filters to both handlers
    handler.addFilter(NoiseFilter())
    console.addFilter(NoiseFilter())

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


def _listener_process(queue, config, log_file_prefix, safedir):
    """ Target function for the logging listener process.
    Ignores SIGINT and runs QueueListener for async logging.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Configure the listener process
    logging.Formatter.converter = time.gmtime
    _listener_configurer(config, log_file_prefix, safedir)

    # Start queue listener
    main_logger = logging.getLogger()
    queue_listener = logging.handlers.QueueListener(queue, *main_logger.handlers)
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

def initLogging(config, log_file_prefix="", safedir=None, level=logging.DEBUG):
    """ Called once in the MAIN process (e.g. StartCapture.py). 
    Spawns the listener process and configures logging.

    Arguments:
        config: [object] RMS config object
        log_file_prefix: [str] Optional prefix for log filenames
        safedir: [str] Fallback directory if normal log_path is unwritable
        level: [int] Logging level for the main logger (defaults to DEBUG)
    """
    global _rms_logging_queue, _rms_listener_process, _rms_logger_initialized
    with _rms_init_lock:
        if _rms_logger_initialized:
            return

    # Remove the default handler if it exists
    main_logger = logging.getLogger()
    for handler in main_logger.handlers[:]:  # [:] makes a copy of the list
        main_logger.removeHandler(handler)
            
    # Create logging infrastructure
    _rms_logging_queue = multiprocessing.Queue(-1)
    _rms_listener_process = multiprocessing.Process(
        target=_listener_process,
        args=(_rms_logging_queue, config, log_file_prefix, safedir)
    )
    _rms_listener_process.daemon = True  # Set daemon attribute manually (for Python 2 compatibility)
    _rms_listener_process.start()

    # Set DEBUG on root logger in main process
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG) # Keep root permissive

    # Configure queue handler for main process
    qh = logging.handlers.QueueHandler(_rms_logging_queue)
    qh.setFormatter(logging.Formatter('%(message)s'))
    
    # Set up root logger with queue handler
    main_logger.handlers = []
    main_logger.addHandler(qh)

    # Redirect standard streams
    sys.stderr = LoggerWriter(main_logger, logging.WARNING)
    if config.log_stdout:
        sys.stdout = LoggerWriter(main_logger, logging.INFO)

    main_logger.propagate = False
    _rms_logger_initialized = True
    main_logger.debug("initLogging completed; queue listener started.")
    atexit.register(shutdownLogging)


def shutdownLogging():
    """ Handles cleanup of logging resources.
    Stops the listener process and resets the logging state.
    """
    global _rms_logging_queue, _rms_listener_process, _rms_logger_initialized
    with _rms_init_lock:
        if not _rms_logger_initialized:
            return
        
        # Stop the listener process
        if _rms_listener_process and _rms_listener_process.is_alive():
            _rms_logging_queue.put(None)  # Sentinel
            _rms_listener_process.join(timeout=5)
            if _rms_listener_process.is_alive():
                _rms_listener_process.terminate()
        
        _rms_logger_initialized = False


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
