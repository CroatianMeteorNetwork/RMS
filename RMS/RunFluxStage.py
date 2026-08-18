""" Entry point for the flux preparation stage, run as a separate process.

The stage builds the night's flux products: sensor characterization, collecting areas,
cloud detection and the dense star-scoring chain behind it. It is a batch step - it reads
FF and CALSTARS files and writes its products to disk, returning nothing to its caller -
so it has no reason to run inside the long-lived capture parent.

Running it here, in a process that exits when the stage is done, is what keeps the capture
parent small. Every page the stage touches goes back to the kernel on exit, whether it was
released by Python or merely freed into an allocator arena that never shrinks. The same
exit also bounds the blast radius: a stage that dies takes one night's flux products with
it instead of the station's capture loop (see the OOM notes in Utils.Flux).

The parent hands over its in-memory state through a pickle rather than letting this process
re-read the config, mask and platepar from disk. Those objects are not the ones on disk:
processNight zeroes platepar.vignetting_coeff in memory when a flat is in use, after the
platepar has already been written out, so a reload would silently score the night with a
different vignetting correction.
"""

from __future__ import print_function, division, absolute_import

import argparse
import os
import sys

from RMS.Logger import getLogger
from RMS.Misc import setMultiprocessingStartMethod
from RMS.Pickling import loadPickle


# Name of the pickle the parent writes into the night directory to hand its in-memory
# config, mask and platepar to this process. Kept here so both sides share one definition.
STATE_FILE_NAME = "flux_stage_state.pickle"


def runFluxStageFromState(dir_path, ftpdetectinfo_path, state_dir=None):
    """ Load the parent's handoff state and run the flux preparation stage.

    Arguments:
        dir_path: [str] Path to the night directory.
        ftpdetectinfo_path: [str] Path to the FTPdetectinfo file.

    Keyword arguments:
        state_dir: [str] Directory holding the handoff pickle. The night directory by
            default.
    """

    if state_dir is None:
        state_dir = dir_path

    state = loadPickle(state_dir, STATE_FILE_NAME)

    if not isinstance(state, dict):
        raise IOError("The flux stage handoff state could not be read from {:s}".format(
            os.path.join(state_dir, STATE_FILE_NAME)))

    # Imported here, not at module scope, so that a failure to read the handoff state is
    # reported before the heavy flux stack is pulled in
    from Utils.Flux import prepareFluxFiles

    prepareFluxFiles(state["config"], dir_path, ftpdetectinfo_path,
                     mask=state.get("mask"), platepar=state.get("platepar"),
                     allow_model_fit=state.get("allow_model_fit", True))


if __name__ == "__main__":

    setMultiprocessingStartMethod()

    arg_parser = argparse.ArgumentParser(
        description="Run the flux preparation stage for one night directory.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str,
        help='Path to the night directory with FF files.')

    arg_parser.add_argument('ftpdetectinfo_path', metavar='FTPDETECTINFO_PATH', type=str,
        help='Path to the FTPdetectinfo file for that night.')

    arg_parser.add_argument('--state-dir', metavar='STATE_DIR', type=str, default=None,
        help="Directory holding the handoff pickle written by the parent. The night "
             "directory by default.")

    cml_args = arg_parser.parse_args()

    # Log to stdout only. The parent forwards these lines into the station log, so the
    # night keeps a single log file with a single writer - the invariant the rest of the
    # capture pipeline maintains by funnelling its children through a logging queue.
    log = getLogger("rmslogger", stdout=True)

    runFluxStageFromState(cml_args.dir_path, cml_args.ftpdetectinfo_path,
                          state_dir=cml_args.state_dir)
