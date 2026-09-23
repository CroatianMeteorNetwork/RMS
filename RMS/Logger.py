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
class LoggerWriter:
    """ Used to redirect stdout/stderr to the log.
    """
    def __init__(self, logger, level, stdout_captured=True):
        self.logger = logger
        self.level = level
        self.stdout_captured = stdout_captured

    def write(self, message):
        try:
            if message.strip():
                self.logger.log(self.level, message.strip())
        except Exception:
            if not self.stdout_captured:
                print(f'error during logging to stdout/stderr: attempted to log - {message}')
            else:
                # if stdout is also captured we can't emit any messages 
                pass

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


class EarlyRecordBuffer(logging.Handler):
    """
    Keep log records emitted before initLogging so they can be replayed into the night log.

    Every entry point loads the config before it initializes logging, so warnings raised while
    parsing (clamped FPS, bad binning factor, upload disabled on the default station code) would
    otherwise only ever reach the console and never the uploaded night log.
    """

    def __init__(self, capacity=200):
        logging.Handler.__init__(self)

        self.capacity = capacity
        self.records = []
        self.dropped = 0


    def emit(self, record):

        # A full buffer just stops collecting - a handler must never raise
        if len(self.records) < self.capacity:
            self.records.append(record)

        else:
            self.dropped += 1


class _SuppressReplayedOnConsole(logging.Filter):
    """
    Drop records replayed from the early buffer so the console does not show a config warning
    twice. Such a record was already printed to stderr by _default_handler before logging was
    initialized; the replay exists only to get it into the night-log FILE. Applied to the console
    handler only - the file handler still receives the replayed record.
    """

    def filter(self, record):
        return not getattr(record, "replayed_from_early_buffer", False)


def installEarlyLogBuffer(level=logging.WARNING):
    """
    Attach a record buffer to the root logger so records emitted before initLogging can be
    replayed into the night log. Meant to be called at the start of config parsing (see
    ConfigReader.parse) - the one path every entry point hits before it initializes logging.

    Idempotent and self-managing:
      - a second call while still pre-init returns the existing buffer,
      - once real logging is up (a QueueHandler on the root logger, installed by initLogging or
        initChildProcess) it is a no-op, since records already reach the night log directly and a
        buffer installed now would never be drained.

    The console side is already covered by _default_handler, installed on the root logger when
    this module is imported. Only warnings and above are kept by default, so the buffer cannot
    fill up with third-party chatter before the interesting records arrive.

    Return:
        [EarlyRecordBuffer or None] The buffer on the root logger, or None if real logging is
            already initialized.
    """

    root = logging.getLogger()

    # Real logging already up: records reach the night log directly, so a buffer installed now
    # would only accumulate undrained. No-op.
    for handler in root.handlers:
        if isinstance(handler, logging.handlers.QueueHandler):
            return None

    # Pre-init: reuse a buffer if one is already attached
    for handler in root.handlers:
        if isinstance(handler, EarlyRecordBuffer):
            return handler

    buffer_handler = EarlyRecordBuffer()
    buffer_handler.setLevel(level)
    root.addHandler(buffer_handler)

    return buffer_handler


def _drainEarlyLogBuffer():
    """
    Take the records off any early buffer on the root logger.

    Return:
        [tuple] (records, dropped) - the buffered records and the number that did not fit.
    """

    root = logging.getLogger()

    records = []
    dropped = 0

    for handler in root.handlers:
        if isinstance(handler, EarlyRecordBuffer):
            records.extend(handler.records)
            dropped += handler.dropped
            handler.records = []
            handler.dropped = 0

    return records, dropped


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
    logger = logging.getLogger("rmslogger") 
    
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
        self.sequence_number = 1  # Initialize sequence counter

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

        # Increment sequence number for the new file
        self.sequence_number += 1

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
            "{}log_{}_{}_{:03d}.log".format(self.log_file_prefix, self.station_id, new_time_str, self.sequence_number)
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



# Upper bound on records buffered between producers and the listener. A stalled listener pipeline
# otherwise grows every producer's queue feeder buffer without bound (observed as ~700 MB/process/day
# in production when a process died while holding the queue's shared write lock)
LOG_QUEUE_MAXSIZE = 30000

# Backlog size above which the listener pipeline is considered stalled
LOG_QUEUE_STALL_THRESHOLD = 500

# Number of consecutive health checks the backlog must persist before the pipeline is restarted
LOG_STALL_STRIKES = 3

# Minimum spacing between two strike-eligible samples. Both the capture watchdog and the
# manager's own health thread call the check every 60 s, so without this the samples can land
# seconds apart and all LOG_STALL_STRIKES accumulate inside a single burst
LOG_STALL_SAMPLE_INTERVAL = 45

# Exponential backoff between pipeline restarts (a persistent failure - e.g. an
# unwritable log directory - must not leak an abandoned queue's fds every 60 s)
LOG_RESTART_BACKOFF_BASE = 120
LOG_RESTART_BACKOFF_MAX = 3600

# Interval of the manager's own always-on health thread
LOG_HEALTH_INTERVAL = 60

# The most recently initialized LoggingManager, used by the module-level health check
_active_manager = None


class _DroppingQueueHandler(logging.handlers.QueueHandler):
    """ QueueHandler that silently drops records when the queue is full or broken.

    The logging queue is bounded, so a stalled listener cannot grow producer memory without
    bound. Blocking or raising here would stall capture, so dropping is the only safe option.
    """
    dropped = 0   # per-process: each forked producer counts its own drops

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except Exception:
            self.dropped += 1
            return

        # Report drops as soon as the queue has room again: forked children never run the
        # health check, so without this their drop counts would stay invisible while the
        # children were the heavy producers (review finding)
        if self.dropped:
            n, self.dropped = self.dropped, 0
            report = logging.LogRecord(
                'rmslogger', logging.WARNING, __file__, 0,
                '{:d} log records from process {:d} were dropped while the '
                'logging queue was full'.format(n, os.getpid()), None, None)
            try:
                self.queue.put_nowait(report)
            except Exception:
                self.dropped += n

    def takeDroppedCount(self):
        """ Return and reset this process's dropped-record count. """
        n, self.dropped = self.dropped, 0
        return n


class LoggingManager:
    """Manages the lifecycle of the multiprocessing logger."""
    def __init__(self):
        self.logging_queue = None
        self.listener_process = None
        self.is_initialized = False

        # The health thread holds this for the duration of every periodic check, so it can
        # be held at fork time - a child inheriting it locked would deadlock on any
        # LoggingManager call. Nothing forked calls one today (every initLogging call lives
        # in a __main__ block); keep it that way, or fork before initLogging.
        self.init_lock = threading.Lock()

        # Arguments used to spawn the listener, kept so a wedged listener can be respawned
        self._listener_args = None

        # Consecutive health checks with an excessive backlog and no listener progress
        self._stall_strikes = 0
        self._last_processed = None
        self._last_sample_time = None

        # Records the current listener has taken off the queue (lock-free, single writer)
        self._records_processed = None

        # Restart pacing (see checkLoggingHealth)
        self._restart_count = 0
        self._next_restart_allowed = 0.0

        # Always-on health thread (started by initLogging)
        self._health_stop = None
        self._health_thread = None

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

        # Helper function to get logging level from a string
        def get_log_level(level_str, default=logging.INFO):
            level_map = {
                "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
                "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL
            }
            return level_map.get(str(level_str).upper(), default)

        # If levels are not passed as arguments, get them from the config file
        if console_level is None:
            console_level = get_log_level(config.console_log_level, logging.INFO)
        
        if file_level is None:
            file_level = get_log_level(config.log_file_log_level, logging.DEBUG)


        with self.init_lock:
            if self.is_initialized:
                return

            # Take anything logged before this point off the early buffer, so it can be replayed
            # into the night log once the queue handler is up. Config parsing happens before
            # initLogging in every entry point, so this is where its warnings come from.
            early_records, early_dropped = _drainEarlyLogBuffer()

            # Remove any default handlers from the root logger
            main_logger = logging.getLogger()
            for handler in main_logger.handlers[:]:
                main_logger.removeHandler(handler)

            # Spawn the listener process
            self._listener_args = (config, log_file_prefix, safedir, console_level, file_level)
            self._spawnListener()

            # Configure the queue handler in the main process
            qh = _DroppingQueueHandler(self.logging_queue)
            qh.setFormatter(logging.Formatter('%(message)s'))
            qh.addFilter(InRmsFilter(config))

            # Replace root handlers with our queue handler
            main_logger.handlers = [qh]
            main_logger.setLevel(min(console_level, file_level)) # Keep root logger permissive
            main_logger.propagate = False

            # Replay the pre-init records into the night log. handle() skips the logger level
            # check but still runs the handler filters, so InRmsFilter drops non-RMS records as
            # usual, and each record keeps its original timestamp. Mark them so the listener's
            # console handler can skip them: _default_handler already printed them to stderr
            # before logging came up, so replaying to the console too would double them - the
            # replay exists only to get them into the night-log file.
            for record in early_records:
                record.replayed_from_early_buffer = True
                main_logger.handle(record)

            if early_dropped:
                main_logger.warning(
                    "{:d} log record(s) emitted before logging was initialized were dropped".format(
                        early_dropped))

            # Redirect standard streams
            sys.stderr = LoggerWriter(main_logger, logging.WARNING, stdout_captured=config.log_stdout)
            if config.log_stdout:
                sys.stdout = LoggerWriter(main_logger, logging.INFO, stdout_captured=config.log_stdout)

            self.is_initialized = True
            main_logger.debug("initLogging completed; queue listener started.")

            # Make this manager visible to the module-level health check
            global _active_manager
            _active_manager = self

            # Always-on health monitoring: the capture watchdog only runs
            # during capture, leaving the multi-hour processing/upload phases
            # unmonitored - a stall there stayed log-dark until the next
            # session (review finding). A daemon thread covers the whole
            # process lifetime; the watchdog's explicit calls stay harmless.
            self._health_stop = threading.Event()
            self._health_thread = threading.Thread(target=self._healthLoop,
                name='rms-log-health')
            self._health_thread.daemon = True
            self._health_thread.start()

            # Register the instance's shutdown method for clean exit
            atexit.register(self.shutdownLogging)


    def _spawnListener(self):
        """ Create a fresh (bounded) logging queue and start a listener process draining it. """

        self.logging_queue = multiprocessing.Queue(LOG_QUEUE_MAXSIZE)

        # Progress counter for the health check. Created fresh with every listener, so a
        # restart starts from zero. lock=False: the listener is the only writer and the health
        # check only ever compares it against its own previous reading, so a locked Value
        # would add a semaphore a dying listener could orphan for no benefit
        self._records_processed = multiprocessing.Value('l', 0, lock=False)

        self.listener_process = multiprocessing.Process(
            target=_listener_process,
            args=(self.logging_queue, self._records_processed) + self._listener_args
        )
        self.listener_process.daemon = True
        self.listener_process.start()


    def checkLoggingHealth(self):
        """ Detect a dead or starved logging pipeline and restart it.

        Two failure modes are covered:
        - The listener process died (crash, OOM kill): trivially detectable.
        - The listener is alive but records stopped reaching it. This happens when any
          producer process dies (or is killed) while holding one of the queue's shared
          internal locks - every producer then buffers records forever and the station
          goes silently log-dark while continuing to run.

        Call periodically from the main process (e.g. the capture watchdog).

        Return:
            [bool] True if the pipeline is healthy, False if a restart was triggered.
        """

        with self.init_lock:

            if not self.is_initialized:
                return True

            # Report records this process dropped while the queue was full -
            # loss must be visible, not silent (per-process count; forked
            # children count their own)
            for handler in logging.getLogger().handlers:
                if isinstance(handler, _DroppingQueueHandler):
                    n_dropped = handler.takeDroppedCount()
                    if n_dropped:
                        logging.getLogger("rmslogger").warning(
                            '%d log records were dropped while the logging '
                            'queue was full', n_dropped)

            # Restart pacing: a persistent failure (unwritable log dir, ...)
            # restarts on an exponential backoff instead of leaking an
            # abandoned queue every minute
            now = time.monotonic()

            # Listener process died
            if (self.listener_process is None) or (not self.listener_process.is_alive()):
                if now < self._next_restart_allowed:
                    print('LOGGING WATCHDOG: listener dead; next restart '
                          'attempt in {:.0f} s'.format(
                          self._next_restart_allowed - now),
                          file=sys.__stderr__)
                    return False
                self._restartLogging('listener process is dead')
                return False

            # A long stretch of health clears the restart pacing
            if self._restart_count and (now > self._next_restart_allowed
                    + LOG_RESTART_BACKOFF_MAX):
                self._restart_count = 0

            # Space the samples out: this check has two callers (the capture
            # watchdog and the always-on health thread), each on its own 60 s
            # cycle, so back-to-back samples are possible and three strikes
            # could otherwise be collected within a single burst
            if (self._last_sample_time is not None) \
                    and ((now - self._last_sample_time) < LOG_STALL_SAMPLE_INTERVAL):
                return True

            # Listener alive - check that the queue is actually draining
            try:
                backlog = self.logging_queue.qsize()
            except (NotImplementedError, OSError):
                # qsize() is not implemented on some platforms (e.g. macOS)
                return True

            processed = self._records_processed.value

            # A stall is the listener not CONSUMING - not a large backlog, and not even a
            # backlog that keeps growing: producers can legitimately outrun the listener for
            # a while (a detection burst on a slow SD card), and killing it then discards
            # everything queued. The listener's own progress counter separates the two cases
            # exactly: it stops advancing only when nothing is being taken off the queue.
            if (backlog > LOG_QUEUE_STALL_THRESHOLD) \
                    and (self._last_processed is not None) \
                    and (processed == self._last_processed):
                self._stall_strikes += 1
            else:
                self._stall_strikes = 0

            self._last_processed = processed
            self._last_sample_time = now

            if self._stall_strikes >= LOG_STALL_STRIKES:
                if now < self._next_restart_allowed:
                    print('LOGGING WATCHDOG: backlog stalled; next restart '
                          'attempt in {:.0f} s'.format(
                          self._next_restart_allowed - now),
                          file=sys.__stderr__)
                    return False
                self._restartLogging('backlog of {:d} records with no listener progress in '
                    '{:d} checks'.format(backlog, LOG_STALL_STRIKES))
                return False

            return True

    def _healthLoop(self):
        """ Always-on periodic health check (daemon thread in the process that
            initialized logging). """

        while not self._health_stop.wait(LOG_HEALTH_INTERVAL):
            try:
                self.checkLoggingHealth()
            except Exception as e:
                print('LOGGING WATCHDOG: health check failed: {!r}'.format(e),
                      file=sys.__stderr__)


    def _restartLogging(self, reason):
        """ Replace the queue and listener process. Must be called with init_lock held. """

        # This must bypass the stream redirect entirely: with log_stdout, plain
        # print() goes to a LoggerWriter still pointing at the wedged queue and
        # the diagnostic is silently lost (review finding)
        print('LOGGING WATCHDOG: {:s} - restarting the logging pipeline...'.format(reason),
              file=sys.__stderr__)

        try:
            if (self.listener_process is not None) and self.listener_process.is_alive():
                self.listener_process.terminate()
                self.listener_process.join(timeout=2)
        except Exception as e:
            print('LOGGING WATCHDOG: could not stop the old listener: {}'.format(e),
                  file=sys.__stderr__)

        # Abandon the old queue - it may be wedged beyond recovery. Records buffered in it are
        # lost, and child processes forked before this point keep logging into the abandoned
        # (bounded) queue, so their output is lost until they are restarted. close() asks our
        # end's feeder thread to exit, releasing this process's pipe fds - best-effort only:
        # a feeder blocked on the very lock that wedged the queue never gets the message and
        # keeps its fds, thread and buffered records. The restart backoff is what bounds that
        # leak (review finding); earlier-forked children still hold their own ends.
        old_queue = self.logging_queue
        try:
            old_queue.close()
            old_queue.cancel_join_thread()
        except Exception:
            pass

        self._restart_count += 1
        self._next_restart_allowed = time.monotonic() + min(
            LOG_RESTART_BACKOFF_BASE*(2**(self._restart_count - 1)),
            LOG_RESTART_BACKOFF_MAX)

        self._spawnListener()
        self._stall_strikes = 0
        self._last_processed = None
        self._last_sample_time = None

        # Point this process's queue handlers at the fresh queue
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.handlers.QueueHandler):
                handler.queue = self.logging_queue

        logging.getLogger("rmslogger").warning(
            'Logging pipeline restarted by watchdog ({:s}). Records logged during the stall are lost.'.format(reason))

    def shutdownLogging(self):
        """
        Handles cleanup of logging resources. Stops the listener process.
        """
        if self._health_stop is not None:
            self._health_stop.set()

        with self.init_lock:
            if not self.is_initialized:
                return
            
            # Stop the listener process
            if self.listener_process and self.listener_process.is_alive():
                print("Shutting down logging...") # Added for visibility during shutdown
                # Sentinel to stop the listener loop. Never block on a possibly wedged queue -
                # the terminate() below handles a listener that does not get the sentinel.
                try:
                    self.logging_queue.put_nowait(None)
                except Exception:
                    pass
                self.listener_process.join(timeout=5)
                if self.listener_process.is_alive():
                    self.listener_process.terminate()
            
            self.is_initialized = False
            print("Logging shutdown complete.")


def checkLoggingHealth():
    """ Run a health check on the most recently initialized LoggingManager, if any.

    Return:
        [bool] True if the pipeline is healthy (or logging was never initialized).
    """

    if _active_manager is None:
        return True

    return _active_manager.checkLoggingHealth()


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

    # Generate log filename with timestamp and initial sequence number
    start_time_str = RmsDateTime.utcnow().strftime("%Y%m%d_%H%M%S")
    logfile_name = "{}log_{}_{}_{:03d}.log".format(log_file_prefix, config.stationID, start_time_str, 1)
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
    # sys.__stdout__ is the process's REAL stdout. After a watchdog restart the
    # listener is forked from a process whose sys.stdout is a LoggerWriter -
    # building the console handler on that recurses every record back into the
    # logging pipeline, amplifying it endlessly (review finding).
    console = logging.StreamHandler(sys.__stdout__ or sys.stdout)

    # Set different levels for each handler
    handler.setLevel(file_level)
    console.setLevel(console_level)
    
    # Add filters to both handlers
    handler.addFilter(InRmsFilter(config))
    console.addFilter(InRmsFilter(config))

    # Config warnings replayed from the early buffer were already shown on the console before
    # logging was initialized; keep them in the file but skip them on the console (no double print)
    console.addFilter(_SuppressReplayedOnConsole())

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


def _listener_process(queue, records_processed, config, log_file_prefix, safedir,
                      console_level=logging.INFO, file_level=logging.DEBUG):
    """ Target function for the logging listener process.
    Ignores SIGINT and processes messages in strict FIFO order.

    records_processed is a lock-free shared counter of the records taken off the queue. It is
    the health check's only reliable "the listener is consuming" signal - the queue backlog
    alone cannot distinguish a wedged pipeline from producers outrunning a healthy listener.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Configure the listener process
    logging.Formatter.converter = time.gmtime
    _listener_configurer(config, log_file_prefix, safedir, console_level, file_level)

    main_logger = logging.getLogger()
    handlers = tuple(main_logger.handlers)  # stable snapshot

    # Single consumer: preserves FIFO order
    while True:
        try:
            record = queue.get()

            # Count progress as soon as the record leaves the queue: the backlog the health
            # check reads shrinks at this point, so the two readings stay consistent even if
            # a handler write is slow
            records_processed.value += 1

            if record is None:       # shutdown sentinel
                break
            # Process record through each handler that accepts its level
            for h in handlers:
                if record.levelno >= h.level:
                    h.handle(record)
        except Exception as e:
            print("Error in listener process: {}".format(e))
            continue

    # Flush all handlers on shutdown
    for h in handlers:
        try: 
            h.flush()
        except: 
            pass


##############################################################################
# PUBLIC ENTRY POINT
##############################################################################

def getWhiteAndBlackLists(config):
    """ 
    Returns the whitelisted RMS root and external script directories, and the blacklisted 
    site-packages directories.
    
    Return:
        (set, set) Tuple of whitelisted and blacklisted directories
    """
    # Whitelist RMS root and external script directories
    rms_root = os.path.realpath(getRmsRootDir())
    allowed_dirs = {rms_root}
    ext = config.external_script_path
    if ext:
        ext_root = os.path.realpath(ext)
        if not os.path.isdir(ext_root):
            ext_root = os.path.dirname(ext_root)
        allowed_dirs.add(ext_root)

    # Blacklist site-packages directories (with Py2 fallback)
    try:
        site_packages = site.getsitepackages()
        user_site     = site.getusersitepackages()
    except (AttributeError, IOError):
        from distutils.sysconfig import get_python_lib
        site_packages = [get_python_lib()]
        user_site     = getattr(site, 'USER_SITE', get_python_lib(prefix=sys.prefix))

    site_dirs = set(os.path.realpath(p) for p in site_packages)
    site_dirs.add(os.path.realpath(user_site))

    return allowed_dirs, site_dirs


def getLogger(name=None, level="DEBUG", stdout=False):
    """ Get a logger instance.
    
    Arguments:
        name: [str] Logger name. If None, returns "rmslogger"
        level: [str] Logging level to set ("DEBUG","INFO","WARNING","ERROR","CRITICAL")
        stdout: [bool] If True, adds a StreamHandler to stdout
        
    Return:
        [Logger] Logger instance
    """
    logger = logging.getLogger(name if name else "rmslogger")

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


##############################################################################
# BACKWARD COMPATIBILITY WITH OLD initLogging FUNCTION
##############################################################################

# Create a global instance of LoggingManager for backward compatibility
_global_logging_manager = LoggingManager()

def initLogging(config, log_file_prefix="", safedir=None, level=logging.DEBUG):
    """
    Backward compatibility wrapper for the old initLogging function.
    
    This function maintains the same interface as the old initLogging but delegates
    to the new LoggingManager implementation.
    
    Arguments:
        config: [object] RMS config object
        log_file_prefix: [str] Optional prefix for log filenames
        safedir: [str] Fallback directory if normal log_path is unwritable
        level: [int] Logging level for the main logger (defaults to DEBUG)
    """
    # The old implementation used a single level parameter for both console and file
    # For backward compatibility, we'll use the provided level for both
    console_level = level
    file_level = level
    
    # Check if config has the new log level attributes
    # If so, use them; otherwise fall back to the provided level
    if hasattr(config, 'console_log_level'):
        # The new initLogging will handle parsing these from config
        console_level = None
    
    if hasattr(config, 'log_file_log_level'):
        # The new initLogging will handle parsing these from config
        file_level = None
    
    # Call the new implementation through the global manager instance
    _global_logging_manager.initLogging(
        config=config,
        log_file_prefix=log_file_prefix,
        safedir=safedir,
        console_level=console_level,
        file_level=file_level
    )


##############################################################################
# CHILD PROCESS SETUP (forkserver / spawn compatibility)
##############################################################################

def getLoggingQueue():
    """ Return the multiprocessing logging queue created by initLogging, or None.

    Child Process classes grab this in their (parent-side) __init__ and pass it to
    initChildProcess() inside run(), so their log records still reach the listener
    process. This is required under the 'forkserver'/'spawn' start methods (the default
    on Linux from Python 3.14), where a child does NOT inherit the parent's root logger
    handlers. Under 'fork' the queue is inherited anyway, so this is harmless there.

    Return:
        [multiprocessing.Queue or None] The shared logging queue, or None if logging was
            never initialized in this process.
    """
    # Prefer the global manager's queue, if it was used
    if _global_logging_manager.logging_queue is not None:
        return _global_logging_manager.logging_queue

    # Otherwise read the queue straight from the root logger's QueueHandler. This is the
    # authoritative source and works no matter which LoggingManager instance configured
    # logging - e.g. StartCapture/Reprocess create their own LoggingManager() rather than
    # using the global one, so the global manager's queue stays None.
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.handlers.QueueHandler):
            return handler.queue

    return None


def initChildLogging(logging_queue, config):
    """ Attach a QueueHandler to the root logger inside a child process.

    Under 'fork' children inherit the parent's QueueHandler, but under 'forkserver'/'spawn'
    they start with a fresh logging configuration. Call this (or initChildProcess) at the
    top of a child process's run() so its records are forwarded to the listener process.
    A no-op if logging was never initialized (queue is None).

    Arguments:
        logging_queue: [multiprocessing.Queue] The shared logging queue, or None.
        config: [Config] Config used for the InRmsFilter. If None, no filter is applied.
    """
    if logging_queue is None:
        return

    root = logging.getLogger()

    # Replace any inherited/default handlers with a single queue handler
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    qh = logging.handlers.QueueHandler(logging_queue)
    qh.setFormatter(logging.Formatter('%(message)s'))
    if config is not None:
        qh.addFilter(InRmsFilter(config))

    root.handlers = [qh]
    root.setLevel(logging.DEBUG)  # permissive; level filtering happens in the listener
    root.propagate = False

    # Make the queue discoverable via getLoggingQueue() in this process, so any further
    # child processes spawned from here (e.g. RawFrameSaver from BufferedCapture, Extractor
    # from Compressor) can grab it in their own __init__ under 'forkserver'/'spawn', where
    # the module-level manager state is not inherited.
    _global_logging_manager.logging_queue = logging_queue


def initChildProcess(logging_queue=None, config=None, ignore_sigint=True):
    """ Re-establish logging and signal handling at the start of a child process's run().

    Needed under the 'forkserver'/'spawn' start methods (the default on Linux from Python
    3.14), where a child does NOT inherit the parent's logging handlers or signal
    dispositions. Under 'fork' this simply re-creates the configuration the child would
    have inherited, so it is safe on every supported Python version.

    Arguments:
        logging_queue: [multiprocessing.Queue] Shared logging queue, or None.
        config: [Config] Config for the logging filter, or None.
        ignore_sigint: [bool] If True, ignore SIGINT in the child and let the parent
            coordinate shutdown via the child's exit Event (mirrors the log listener).
    """
    if ignore_sigint:
        import signal
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            # signal() only works in the main thread of the main interpreter
            pass

    initChildLogging(logging_queue, config)
