
""" Runs a function from an external script defined in the config file. """

from __future__ import print_function, division, absolute_import

import os
import sys
import importlib
import multiprocessing
import traceback

import RMS.ConfigReader as cr
from RMS.Logger import getLogger, getLoggingQueue

# Get the logger from the main module
log = getLogger("rmslogger")


def _runExternalInChild(external_script_dir, module_name, function_name, captured_night_dir,
                        archived_night_dir, config, inhibit_logging, logging_queue):
    """ Runs in the child process: re-establishes sys.path, imports the user module and calls
        the requested function.

        Importing the user module inside the child (rather than pickling the already-imported
        function object) is what keeps this working under the 'forkserver'/'spawn' start
        methods (the default on Linux from Python 3.14). There the child is a fresh
        interpreter that does NOT inherit the parent's sys.path, so a function pickled by
        reference (module.func) would fail to import.

    Arguments:
        external_script_dir: [str] Directory containing the user script (added to sys.path).
        module_name: [str] Importable module name of the user script.
        function_name: [str] Name of the function to call within the module.
        captured_night_dir: [str] Path to the Captured night directory.
        archived_night_dir: [str] Path to the Archived night directory.
        config: [Config instance]
        inhibit_logging: [bool] If True, remove inherited log handlers in this process.
        logging_queue: [multiprocessing.Queue] Shared logging queue, or None. Used to
            re-attach logging when it is not inhibited (handlers are not inherited under
            'forkserver'/'spawn').
    """

    if external_script_dir not in sys.path:
        sys.path.insert(0, external_script_dir)

    if inhibit_logging:
        # Remove any (inherited) handlers so the external script does not log to the RMS log
        import logging
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers[:]:
                root.removeHandler(handler)
                handler.close()
    else:
        # Re-attach logging so the external script's records reach the listener, which under
        # 'forkserver'/'spawn' a fresh child does not inherit. Logging only - signal handling
        # is left untouched for the user script.
        from RMS.Logger import initChildLogging
        initChildLogging(logging_queue, config)

    # Import the user module and resolve the function in the child process
    module = importlib.import_module(module_name)
    external_function = getattr(module, function_name)

    # Call the external function
    external_function(captured_night_dir, archived_night_dir, config)



def runExternalScript(captured_night_dir, archived_night_dir, config):
    """ Run the external script. Its results won't be returned to the main program, the script will just be 
        run as a separate process.
    
    Arguments:
        captured_night_dir: [str] Path to the Captured night directory.
        archived_night_dir: [str] Path to the Archived night directory.
        config: [Config instance]

    Return:
        None
    """


    # Check if running the script is enabled
    if not config.external_script_run:
        return None

    if (config.external_script_path is None) or (config.external_function_name is None):
        log.error('To run an external script, both the path to the script and the name of the function to run must be defined in the config file!')
        return None

    # Check if the script path exists
    if not os.path.isfile(os.path.expanduser(config.external_script_path)):
        log.error('The script {:s} does not exist!'.format(config.external_script_path))
        return None


    try:

        # Extract the name of the folder and the script
        external_script_dir, external_script_file = os.path.split(os.path.expanduser(config.external_script_path))
        module_name = external_script_file.replace('.py', '').replace('.PY', '')

        # Insert the path to the script (in the parent, so the validation import below works)
        if external_script_dir not in sys.path:
            sys.path.insert(0, external_script_dir)

        # Validate that the module and function exist before spawning, so any error is logged
        # clearly in the parent. The child re-imports independently (see _runExternalInChild).
        module = importlib.import_module(module_name)
        externalFunction = getattr(module, config.external_function_name)

        # Call the external function in a separate process, protecting the main process from
        # potential crashes. The module name and function name (not the function object) are
        # passed so the child can re-import them under any multiprocessing start method.
        inhibit_logging = not config.external_script_log
        if inhibit_logging:
            log.info('Starting function "{}" from external script "{}" with logging inhibited'.format(externalFunction, module))
        else:
            log.info('Starting function "{}" from external script "{}"'.format(externalFunction, module))

        p = multiprocessing.Process(
            target=_runExternalInChild,
            args=(external_script_dir, module_name, config.external_function_name,
                  captured_night_dir, archived_night_dir, config, inhibit_logging,
                  getLoggingQueue()))
        p.start()

        if config.external_script_log:
            log.info('External script now running as a separate process')


    except Exception as e:
        log.error('Running external script failed with error:' + repr(e))
        log.error(*traceback.format_exception(*sys.exc_info()))




if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Run external script.
        """)

    arg_parser.add_argument('captured_path', nargs=1, metavar='CAPTURED_PATH', type=str, 
        help='Path to Captured night directory.')

    arg_parser.add_argument('archived_path', nargs=1, metavar='ARCHIVED_PATH', type=str, 
        help='Path to Archived night directory.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, 
        help="Path to a config file which will be used instead of the default one.")
    
    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ######
    # Start log to stdout
    log = getLogger("rmslogger", stdout=True)

    # Load config file
    if cml_args.config is None:
        config = cr.parse(".config")
    else:
        pth, cfg = os.path.split(cml_args.config[0])
        config = cr.loadConfigFromDirectory(cfg, pth)

    # Run the external script
    runExternalScript(cml_args.captured_path[0], cml_args.archived_path[0], config)
