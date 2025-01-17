import os
import sys
import errno
import logging
import logging.handlers
import multiprocessing
import datetime
import time
import threading
import atexit


##############################################################################
# GLOBALS
##############################################################################

# Set GStreamer debug level. Use '2' for warnings in production environments.
# Level 4 and above are overwhelming the log
# If higher verbosity is needed, disable in client scripts
os.environ['GST_DEBUG'] = '2'

logging_queue = None
listener_process = None
init_lock = threading.Lock()
logger_initialized = False


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
root = logging.getLogger()
root.addHandler(_default_handler)
root.setLevel(logging.DEBUG)

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
    
    The live file: log_XX0000_2024-12-29_112347.log
    On rollover: log_XX0000_2024-12-29_112347-[29_1123-to-30_1123].log
    """
    def __init__(self, station_id, start_time_str, *args, **kwargs):
        self.station_id = station_id
        self.start_time_str = start_time_str
        super(CustomHandler, self).__init__(*args, **kwargs)
        self.suffix = "%Y-%m-%d_%H%M%S"
        self.namer = self._rename_on_rollover

    def _rename_on_rollover(self, default_name):
        # Parse the default filename
        base_dir, base_file = os.path.split(default_name)
        base_noext, dot, start_time_str = base_file.rpartition('.')
        
        if base_noext.endswith('.log'):
            base_noext = base_noext[:-4]
        
        # Calculate time range for the log file
        start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d_%H%M%S")
        end_time = datetime.datetime.fromtimestamp(self.rolloverAt)
        
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
        self.noisy_modules = {'font_manager', 'ticker', 'transport', 'sftp', 'dvrip', 'channel', 'cmd'}

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
    mkdirP(log_path)

    # Use safedir if main path isn't writable
    if safedir:
        if not os.path.exists(log_path) or not os.access(log_path, os.W_OK):
            root_logger.debug("Log directory not writable, using safedir: %s", safedir)
            log_path = safedir
            mkdirP(log_path)

    # Generate log filename with timestamp
    start_time_str = RmsDateTime.utcnow().strftime("%Y-%m-%d_%H%M%S")
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
    _listener_configurer(config, log_file_prefix, safedir)

    # Start queue listener
    root_logger = logging.getLogger()
    queue_listener = logging.handlers.QueueListener(queue, *root_logger.handlers)
    queue_listener.start()

    # Keep the process alive
    while True:
        time.sleep(60)


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
    global logging_queue, listener_process, logger_initialized
    with init_lock:
        if logger_initialized:
            return

    # Remove the default handler if it exists
    root = logging.getLogger()
    for handler in root.handlers[:]:  # [:] makes a copy of the list
        root.removeHandler(handler)
            
    # Create logging infrastructure
    logging_queue = multiprocessing.Queue(-1)
    listener_process = multiprocessing.Process(
        target=_listener_process,
        args=(logging_queue, config, log_file_prefix, safedir),
        daemon=True
    )
    listener_process.start()

    # Set DEBUG on root logger in main process
    root = logging.getLogger()
    root.setLevel(logging.DEBUG) # Keep root permissive

    # Configure queue handler for main process
    qh = logging.handlers.QueueHandler(logging_queue)
    qh.setFormatter(logging.Formatter('%(message)s'))
    
    # Set up root logger with queue handler
    root.handlers = []
    root.addHandler(qh)

    # Redirect standard streams
    sys.stderr = LoggerWriter(root, logging.WARNING)
    if config.log_stdout:
        sys.stdout = LoggerWriter(root, logging.INFO)

    root.propagate = False
    logger_initialized = True
    root.debug("initLogging completed; queue listener started.")
    atexit.register(shutdownLogging)


def shutdownLogging():
    """ Handles cleanup of logging resources.
    Stops the listener process and resets the logging state.
    """
    global logging_queue, listener_process, logger_initialized
    with init_lock:
        if not logger_initialized:
            return
        
        # Stop the listener process
        if listener_process and listener_process.is_alive():
            logging_queue.put(None)  # Sentinel
            listener_process.join(timeout=5)
            if listener_process.is_alive():
                listener_process.terminate()
        
        logger_initialized = False


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
