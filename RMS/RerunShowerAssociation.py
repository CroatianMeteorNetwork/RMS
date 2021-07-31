import sys
import os
import argparse
import logging
import traceback
import glob


import RMS.ConfigReader as cr
from RMS.Logger import initLogging
from Utils.ShowerAssociation import showerAssociation


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Re-running the shower association for testing.
        """)

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to the folder with FF files.')    

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ######


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Initialize the logger
    initLogging(config)

    # Get the logger handle
    log = logging.getLogger("logger")

    # Perform single station shower association
    log.info("Performing single station shower association...")

    data_dir = cml_args.dir_path[0]
    log.info(f"data dir: {data_dir}")

    FTPdetectinfo_files = glob.glob('{:s}/FTPdetectinfo_*.txt'.format(data_dir))
    ftpdetectinfo_name = None
    for f in FTPdetectinfo_files:
        if 'backup' not in f:
            ftpdetectinfo_name = os.path.basename(f)

    if not ftpdetectinfo_name:
        log.error("FTPdetectinfo_*.txt file not found, giving up")
    else:
        try:
            showerAssociation(config, [os.path.join(data_dir, ftpdetectinfo_name)], \
                save_plot=True, plot_activity=True)

        except Exception as e:
            log.debug('Shower association failed with the message:\n' + repr(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))
