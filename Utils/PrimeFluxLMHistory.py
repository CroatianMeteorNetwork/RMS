""" Prime the per-camera flux LM history from already-processed nights.

The empirical limiting-magnitude correction (see Utils.Flux.empiricalLMCorrection) needs
several nights of history before it activates. Stations typically have months of archived
nights processed by earlier RMS versions - each with a platepars_flux_recalibrated.json that
contains everything the history needs (matched star lists and per-FF photometric
zero-points), measured the same way detectClouds does at runtime. This utility walks those
archives in chronological order and replays them through the same code path the nightly
pipeline uses, so a station gets its correction on the first night after deployment instead
of a week later. Nights predating flux recalibration fall back to platepars_all_recalibrated.

Usage:
    python -m Utils.PrimeFluxLMHistory /path/to/ArchivedFiles [--config /path/to/.config]

Note: the model LM uses the CURRENT config's intensity_threshold - archives do not record
the capture-time value. If the threshold was retuned since the archived nights were
captured, prime only from nights captured under the current settings.
"""

from __future__ import print_function, division, absolute_import

import argparse
import json
import os
import types

import numpy as np

import RMS.ConfigReader as cr
from Utils.Flux import (measureNightMatchedDepth, empiricalLMCorrection, stellarLMModel,
                        LM_HISTORY_FILE)


def primeFluxLMHistory(config, archive_dir):
    """ Replay archived nights into the station's flux LM history file.

    Arguments:
        config: [Config] Station config (stationID selects the night directories,
            data_dir hosts the history file, intensity_threshold enters the model LM).
        archive_dir: [str] Directory containing per-night folders (e.g. ArchivedFiles).

    Return:
        n_primed: [int] Number of nights that contributed a history entry.
    """

    # Pinned to the historical network reference (threshold 18 -> -1.2 mag): the
    # candidate gate is noise-adaptive and config.intensity_threshold no longer
    # describes the detector (see Utils.Flux for the same pin)
    star_det_mag_corr = -1.2

    night_dirs = sorted(
        d for d in os.listdir(archive_dir)
        if d.startswith(str(config.stationID) + "_")
        and os.path.isdir(os.path.join(archive_dir, d))
    )

    n_primed = 0
    correction = 0.0
    for night in night_dirs:

        # Measure depth from the FLUX platepars, exactly as detectClouds does at runtime.
        # The flux recalibration matches with the max-star cap lifted, so it reaches deeper
        # than the standard platepars_all_recalibrated (by up to ~0.8 mag on star-rich
        # cameras). Priming from the standard file would seed a too-shallow envelope and
        # over-correct. Fall back to the standard file only for nights processed before flux
        # recalibration existed.
        pp_path = os.path.join(archive_dir, night, "platepars_flux_recalibrated.json")
        if not os.path.isfile(pp_path):
            pp_path = os.path.join(archive_dir, night, "platepars_all_recalibrated.json")
        if not os.path.isfile(pp_path):
            continue

        try:
            with open(pp_path) as f:
                pp_dicts = json.load(f)
        except Exception as e:
            print("  {}: unreadable recalibrated platepars ({})".format(night, e))
            continue

        # Lightweight platepar stand-ins - measureNightMatchedDepth only reads star_list
        pps = {ff: types.SimpleNamespace(star_list=d.get("star_list"))
               for ff, d in pp_dicts.items()}

        depth = measureNightMatchedDepth(pps)
        model_lms = [stellarLMModel(d["mag_lev"]) + star_det_mag_corr
                     for d in pp_dicts.values()
                     if d.get("auto_recalibrated") and ("mag_lev" in d)]

        if (depth is None) or (not model_lms):
            print("  {}: no usable depth measurement, skipped".format(night))
            continue

        model_lm = float(np.median(model_lms))

        # Same entry point as the nightly pipeline: appends the night and returns the
        # correction the history would have applied BEFORE this night
        correction = empiricalLMCorrection(config, night, model_lm, depth)
        n_primed += 1
        print("  {}: model LM {:.2f}, measured depth {:.2f} "
              "(correction before this night: {:.2f})".format(night, model_lm, depth, correction))

    history_path = os.path.join(os.path.expanduser(config.data_dir),
                                "{:s}_{:s}".format(str(config.stationID), LM_HISTORY_FILE))
    print()
    print("Primed {:d} night(s) into {:s}".format(n_primed, history_path))

    return n_primed


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Prime the per-camera flux LM history from archived nights.")
    parser.add_argument("archive_dir", type=str,
        help="Directory with per-night folders (e.g. ~/RMS_data/ArchivedFiles).")
    parser.add_argument("--config", type=str, default=None,
        help="Path to the station config (default: standard config resolution).")

    args = parser.parse_args()

    config = cr.loadConfigFromDirectory(args.config if args.config else '.',
                                        os.path.abspath(args.archive_dir))

    primeFluxLMHistory(config, os.path.abspath(args.archive_dir))
