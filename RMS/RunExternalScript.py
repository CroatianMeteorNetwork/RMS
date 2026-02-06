
""" Runs a function from an external script defined in the config file. """

from __future__ import print_function, division, absolute_import

import os
import sys
import importlib
import multiprocessing
import traceback
import time
import datetime

import RMS.ConfigReader as cr
from RMS.Logger import getLogger

# Get the logger from the main module
log = getLogger("rmslogger")


def externalFunctionWrapper(func, captured_night_dir, archived_night_dir, config):
    """ Wrapper for the external function that removes all log handlers in the external process if needed. """
    
    # Check if logging is loaded
    if 'logging' in sys.modules:
        import logging

        # Remove all handlers from the root logger
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers[:]:
                root.removeHandler(handler)
                handler.close()

    # Call the external function
    func(captured_night_dir, archived_night_dir, config)



def runExternalScript(captured_night_dir, archived_night_dir, config):
    """ Run the external script, or scripts. Their results won't be returned to the main program,
        they will be run as a separate process.
    
    Arguments:
        captured_night_dir: [str] Path to the Captured night directory.
        archived_night_dir: [str] Path to the Archived night directory.
        config: [Config instance]

    Return:
        [dict] : dictionary of external script processes
    """


    # Check if running the script is enabled
    if not config.external_script_run:
        return None

    if (config.external_script_path is None) or (config.external_function_name is None):
        log.error('To run an external script, both the path to the script and the name of the function to run must be defined in the config file!')
        return None

    # Initialise external_script_path_list to hold a list of paths
    # and external script_process_dict to hold a list of process information, key will be PID
    external_script_path_list, external_script_process_dict = [], {}

    if "," in config.external_script_path:
        log.info(f"Found a list of external script paths")
        external_script_path_list = config.external_script_path.split(",")
        for external_script in external_script_path_list:
            log.info(f"                                     {external_script.strip()}")

    else:
        log.info(f"Found a single external script path {config.external_script_path}")
        external_script_path_list = [config.external_script_path]


    for external_script_path in external_script_path_list:

        # Check if the script path exists
        external_script_path = os.path.expanduser(external_script_path.strip())
        if not os.path.isfile(external_script_path):
            log.error('The script {:s} does not exist!'.format(external_script_path))
            continue
        time.sleep(1)

        try:

            # Extract the name of the folder and the script
            external_script_dir, external_script_file = os.path.split(os.path.expanduser(external_script_path))

            # Insert the path to the script
            sys.path.insert(0, external_script_dir)

            # Import the function from the external script
            module = importlib.import_module(external_script_file.replace('.py', '').replace('.PY', ''))
            externalFunction = getattr(module, config.external_function_name)

            # Call the external function in a separate process, protecting the main process from potential crashes

            # If logging is disabled, create a wrapper function which removes all log handlers in the external
            # process
            if not config.external_script_log:
                # Use the wrapper function
                log.info('Starting function "{}" from external script "{}" with logging inhibited'.format(externalFunction, module))
                target_function = externalFunctionWrapper
                args = (externalFunction, captured_night_dir, archived_night_dir, config)

            else:
                log.info('Starting function "{}" from external script "{}"'.format(externalFunction, module))
                target_function = externalFunction
                args = (captured_night_dir, archived_night_dir, config)


            p = multiprocessing.Process(target=target_function, args=args)
            p.start()
            parent_pid = os.getpid()
            log.info(f"Process {p.pid} parent recorded as {parent_pid} running {external_script_path}")
            external_script_process_dict[p.pid] =  {"process" : p,
                                                    "parent_pid" : parent_pid,
                                                    "external_script_path" : external_script_path,
                                                    "start_time" : datetime.datetime.now()}


            if config.external_script_log:
                log.info(f'{module} now running as a separate process with PID {p.pid}')


        except Exception as e:
            log.error('Running external script failed with error:' + repr(e))
            log.error(*traceback.format_exception(*sys.exc_info()))

    return external_script_process_dict

def checkExternalProcesses(external_script_process_dict):
    """ Check the dictionary of external script processes to see if any are still running. Return
    an updated dictionary of running external script processes, and a dictionary of processes which were
    in the external_script_process_dict, but have stopped.

    Processes are tracked by PID, and parent, which makes PID reuse errors quite unlikely.

    Arguments:
        external_script_process_dict: [dict] Dictionary of external script processes.

    Return:
        [dict] : Dictionary of external script processes which are running.
        [dict] : Dictionary of external script processes which were passed in, but have stopped.
    """

    if external_script_process_dict is None:
        return external_script_process_dict, None

    external_script_stopped_process_dict = {}
    for external_script_process_key in external_script_process_dict:
        process_dict = external_script_process_dict[external_script_process_key]
        p = process_dict["process"]
        process_parent = process_dict["parent_pid"]
        if p.is_alive() and process_parent == os.getpid():
            pass
        else:
            external_script_stopped_process_dict[external_script_process_key] = process_dict

    for external_script_process_key in external_script_stopped_process_dict:
        del external_script_process_dict[external_script_process_key]

    return external_script_process_dict, external_script_stopped_process_dict

def runningExternalScripts(external_script_process_dict):
    """
    Given a dictionary of external processes, are any still running.

    Arguments:
        external_script_process_dict: [dict] Dictionary of external script processes.

    Return:
        [bool]: True if any external scripts are still running.
    """
    external_script_process_dict, external_script_stopped_process_dict = \
        checkExternalProcesses(external_script_process_dict)
    if external_script_process_dict is None:
        return False

    if len(external_script_stopped_process_dict):
        return True
    else:
        return False

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
    running_external_process_dict =  runExternalScript(cml_args.captured_path[0], cml_args.archived_path[0], config)
    time.sleep(10)

    if running_external_process_dict is not None:

        while len(running_external_process_dict):
            running_external_process_dict, stopped_external_process_dict = checkExternalProcesses(running_external_process_dict)
            if len(running_external_process_dict) == 1:
                log.info(f"{len(running_external_process_dict)} external script is still running")
            else:
                log.info(f"{len(running_external_process_dict)} external scripts are still running")
            for running_pid in running_external_process_dict:

                log.info(f"  Path       : {running_external_process_dict[running_pid]['external_script_path']}")
                log.info(f"  Parent     : {running_external_process_dict[running_pid]['parent_pid']}")
                start_time = running_external_process_dict[running_pid]['start_time'].strftime("%H:%M:%S")
                run_duration = datetime.datetime.now() - running_external_process_dict[running_pid]['start_time']
                log.info(f"  Started    : {start_time}")
                log.info(f"  Duration   : {run_duration - datetime.timedelta(microseconds=run_duration.microseconds)}")
                log.info("")

            if len(stopped_external_process_dict):
                if len(stopped_external_process_dict) == 1:
                    log.info(f"{len(stopped_external_process_dict)} external script stopped since the last check")
                else:
                    log.info(f"{len(stopped_external_process_dict)} external scripts stopped since the last check")


            for stopped_pid in stopped_external_process_dict:
                log.info(f"  Path       : {stopped_external_process_dict[stopped_pid]['external_script_path']}")
                log.info(f"  Parent     : {stopped_external_process_dict[stopped_pid]['parent_pid']}")
                start_time = stopped_external_process_dict[stopped_pid]['start_time'].strftime("%H:%M:%S")
                log.info(f"  Started    : {start_time}")
                log.info("")

            # Before sleep, check if all the external running processes have completed; so we do not wait without reason
            if not len(running_external_process_dict):
                break
            time.sleep(30)