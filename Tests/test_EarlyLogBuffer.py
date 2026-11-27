""" Tests of the early log buffer: warnings logged while the config is parsed, before logging is initialized,
    must end up in the night log file once, and on the console only once.
"""

from __future__ import print_function, division, absolute_import

import glob
import logging
import logging.handlers
import os
import subprocess
import sys
import textwrap


# Root of the RMS repository (the parent of the Tests directory)
RMS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def _rootHandlers():
    """ Return a copy of the handlers on the root logger, to restore them after a test. """

    return list(logging.getLogger().handlers)


def _restoreRootHandlers(handlers):
    """ Put back the given handlers on the root logger, removing any added by a test.

    Arguments:
        handlers: [list] Handlers returned by _rootHandlers.
    """

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    for handler in handlers:
        root.addHandler(handler)



def testInstallIsIdempotentBeforeLogging():
    """ A second install before logging is initialized reuses the first buffer. """

    from RMS.Logger import installEarlyLogBuffer, EarlyRecordBuffer

    saved = _rootHandlers()
    try:

        # Start from a root logger without a queue handler or a buffer, as before initLogging
        root = logging.getLogger()
        for handler in root.handlers[:]:
            if isinstance(handler, (logging.handlers.QueueHandler, EarlyRecordBuffer)):
                root.removeHandler(handler)

        first = installEarlyLogBuffer()
        second = installEarlyLogBuffer()

        assert first is not None
        assert first is second
        assert sum(isinstance(h, EarlyRecordBuffer) for h in root.handlers) == 1

    finally:
        _restoreRootHandlers(saved)


def testInstallIsNoOpOnceLoggingIsUp():
    """ With a queue handler on the root logger (logging initialized), no buffer is installed. """

    from RMS.Logger import installEarlyLogBuffer, EarlyRecordBuffer

    saved = _rootHandlers()
    try:

        # Logging is up: the root logger feeds a queue
        root = logging.getLogger()
        for handler in root.handlers[:]:
            if isinstance(handler, EarlyRecordBuffer):
                root.removeHandler(handler)
        root.addHandler(logging.handlers.QueueHandler(None))

        assert installEarlyLogBuffer() is None
        assert not any(isinstance(h, EarlyRecordBuffer) for h in root.handlers)

    finally:
        _restoreRootHandlers(saved)


def testFullBufferCountsDroppedRecords():
    """ A full buffer stops keeping records and counts the ones it drops, without raising. """

    from RMS.Logger import EarlyRecordBuffer

    buffer_handler = EarlyRecordBuffer(capacity=3)

    # Log more records than the buffer holds
    for i in range(5):
        record = logging.LogRecord("RMS.test", logging.WARNING, __file__, 1, "warning {:d}".format(i), None,
                                   None)
        buffer_handler.handle(record)

    assert [r.getMessage() for r in buffer_handler.records] == ["warning 0", "warning 1", "warning 2"]
    assert buffer_handler.dropped == 2


def testConfigWarningReachesNightLogOnce(tmp_path):
    """ A warning logged while the config is parsed (the default station code disables the upload) is written
        to the night log file once, and printed on the console once, not twice.
    """

    # Run in a fresh interpreter, so the root logger is as it is at the start of a station run
    script = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, {rms_root!r})
        import RMS.ConfigReader as cr
        from RMS.Logger import LoggingManager

        # Parse the config before logging is up, as every entry point does
        config = cr.parse(os.path.join({rms_root!r}, '.config'))
        config.data_dir = {data_dir!r}
        config.log_dir = 'logs'

        manager = LoggingManager()
        manager.initLogging(config, 'test_')
        manager.shutdownLogging()
    """).format(rms_root=RMS_ROOT, data_dir=str(tmp_path))

    result = subprocess.run([sys.executable, "-c", script], cwd=RMS_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    assert result.returncode == 0, result.stderr

    message = "Disabled upload because the default station code is used!"

    # The warning is in the night log, once
    log_files = glob.glob(os.path.join(str(tmp_path), "logs", "test_log_*.log"))
    assert len(log_files) == 1

    with open(log_files[0]) as f:
        log_text = f.read()

    assert log_text.count(message) == 1

    # The console shows it once (printed before logging came up), not again when it is replayed
    assert (result.stdout + result.stderr).count(message) == 1
