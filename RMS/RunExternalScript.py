
""" Runs a function from an external script defined in the config file. """

from __future__ import print_function, division, absolute_import

import os
import sys
import importlib
import multiprocessing
import traceback
import logging

import RMS.ConfigReader as cr

# Get the logger from the main module
log = logging.getLogger("logger")


def runExternalScript(captured_night_dir, archived_night_dir, config):
    """ Run the external script. It's results won't be returned to the main program, the script will just be run as a separate process.
    
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


    # Check if the script path exists
    if not os.path.isfile(config.external_script_path):
        log.error('The script {:s} does not exist!'.format(config.external_script_path))
        return None


    try:

        # Extract the name of the folder and the script
        external_script_dir, external_script_file = os.path.split(config.external_script_path)

        # Insert the path to the script
        sys.path.insert(0, external_script_dir)

        # Import the function from the external script
        module = importlib.import_module(external_script_file.replace('.py', '').replace('.PY', ''))
        externalFunction = getattr(module, config.external_function_name)

        log.info('Starting function "{}" from external script "{}"'.format(externalFunction, module))

        # Call the external function in a separate process, protecting the main process from potential crashes
        p = multiprocessing.Process(target=externalFunction, args=(captured_night_dir, archived_night_dir, config))
        p.start()

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

    arg_parser.add_argument('captured_path', nargs=1, metavar='CAPTURED_PATH', type=str, \
        help='Path to Captured night directory.')

    arg_parser.add_argument('archived_path', nargs=1, metavar='ARCHIVED_PATH', type=str, \
        help='Path to Archived night directory.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ######
    # Start log to stdout
    log = logging.getLogger("logger")
    out_hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(out_hdlr)

    # Load config file
    config = cr.parse(".config")

    # Run the external script
    runExternalScript(cml_args.captured_path[0], cml_args.archived_path[0], config)
