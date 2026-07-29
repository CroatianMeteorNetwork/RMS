""" Offline fleet replay lab for the transparency system.

The failure modes that cost fleet mornings are STATEFUL: a calibration wiped by
a depth-changing refit needs two nights of accumulated state to exist at all; a
sibling fit race needs concurrent pod processing; a refit wave needs a code
transition mid-corpus. Single-night estimator labs are structurally blind to
them. This lab replays archived station nights IN SEQUENCE against an evolving
lab data_dir - the real detectClouds path end to end (scoring, grid + tree
transparency maps, star-calibration EMA, dome-model staleness/refit) - so a
code change can be judged against a week of real fleet nights in minutes,
before it ships.

Corpus layout (one directory per station, one subdirectory per night, named by
the capture directory):

    <corpus>/<STATION>/<night_name>/
        CALSTARS_<night_name>.txt          (required)
        platepars_flux_recalibrated.json   (required)
        mask.bmp                           (optional)
        .config                            (optional - template + platepar
                                            geo/station fields otherwise)
        <night>_still_star_states.npz      (optional stills sidecar)
        <night>_star_scoring.npz           (reference - station's product)
        <night>_transparency_map*.npz      (reference - station's products)

The corpus is read-only: every replayed night is copied (inputs only) into the
lab run directory. The station's own archived products are never overwritten -
they are the regression reference the replay is diffed against.

Lifecycle invariants checked across nights (violations print as LAB WARNING):
    - the star calibration must accumulate unless the catalog depth changed
    - the installed dome model must carry no quality_issues
    - a replayed night whose code version matches the station's should produce
      a tree map close to the archived one (median |ddm| reported)

Usage:
    python -m Utils.TransparencyReplayLab <corpus_dir> --station CAWEC4 \\
        [--lab <run_dir>] [--fresh]
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import os
import shutil

import numpy as np


# flux_time_intervals.json is deliberately NOT an input: detectClouds treats it
# as a cache and returns the stored intervals without recomputing anything
INPUT_MEMBERS = ("CALSTARS_*.txt", "platepars_flux_recalibrated.json",
    "platepar_cmn2010.cal", "mask.bmp", "*_still_star_states.npz")

REFERENCE_SUFFIXES = ("_star_scoring.npz", "_transparency_map.npz",
    "_transparency_map_tree.npz")


def _labConfig(night_dir, station_id, repo_root):
    """ Station config for the replay: the night's own .config when archived,
        else the repo template with station identity and geo taken from the
        night's platepars. """

    import RMS.ConfigReader as cr

    cfg_path = os.path.join(night_dir, ".config")
    if os.path.isfile(cfg_path):
        config = cr.parse(cfg_path)
    else:
        config = cr.parse(os.path.join(repo_root, ".config"))
        config.stationID = station_id

        pp_path = os.path.join(night_dir, "platepars_flux_recalibrated.json")
        with open(pp_path) as f:
            ppr = json.load(f)
        for v in ppr.values():
            if isinstance(v, dict) and ("lat" in v):
                config.latitude = float(v["lat"])
                config.longitude = float(v["lon"])
                config.elevation = float(v["elev"])
                config.width = int(v.get("X_res", config.width))
                config.height = int(v.get("Y_res", config.height))
                break

    return config


def _calibrationState(state_dir, station_id):
    """ (n_stars_active, max_n_nights, catalog_lim_mag) of the lab calibration
        file, or None if absent. """

    from Utils.StarCalibration import calibrationFileName

    path = os.path.join(state_dir, calibrationFileName(station_id))
    if not os.path.isfile(path):
        return None

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"])) if "header" in z else {}
        n_nights = z["n_nights"] if "n_nights" in z else np.zeros(1)
        return (int(np.sum(n_nights > 0)), int(n_nights.max()) if len(n_nights) else 0,
            float(header.get("catalog_lim_mag", np.nan)))


def _modelState(state_dir, station_id):
    """ (fit_date, depth, issues) of the installed lab dome model, or None. """

    path = os.path.join(state_dir, "{:s}_light_dome.json".format(station_id))
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return (str(d.get("fit_date")), float(d.get("catalog_lim_mag", np.nan)),
        list(d.get("quality_issues") or []))


def _mapDiff(produced_path, reference_path):
    """ Median |ddm| between two transparency maps over cells finite in both,
        or None when either is missing/incompatible. """

    if not (os.path.isfile(produced_path) and os.path.isfile(reference_path)):
        return None
    try:
        with np.load(produced_path, allow_pickle=False) as a, \
                np.load(reference_path, allow_pickle=False) as b:
            da, db = a["dm"], b["dm"]
            if da.shape != db.shape:
                return None
            both = np.isfinite(da) & np.isfinite(db)
            if not np.any(both):
                return None
            return float(np.median(np.abs(da[both] - db[both])))
    except Exception:
        return None


def replayStation(corpus_dir, station_id, lab_dir, fresh=False):
    """ Replay every corpus night of one station chronologically.

    Return:
        report: [list of dict] Per-night state and diff summary.
    """

    from Utils.Flux import detectClouds
    from RMS.Formats.StarScoring import scoringFileName

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    station_corpus = os.path.join(corpus_dir, station_id)
    night_names = sorted(d for d in os.listdir(station_corpus)
        if os.path.isdir(os.path.join(station_corpus, d))
        and d.startswith(station_id))
    if not night_names:
        print("No corpus nights for {:s}".format(station_id))
        return []

    run_dir = os.path.join(lab_dir, station_id)
    state_dir = os.path.join(run_dir, "state")
    if fresh and os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    archived_dir = os.path.join(state_dir, "ArchivedFiles")
    first_run = not os.path.isdir(state_dir)
    os.makedirs(archived_dir, exist_ok=True)

    # Seed the station state (dome model, calibration, histories) from the
    # corpus snapshot, so the replay starts where the real station stood at
    # the corpus start instead of from a multi-night bootstrap
    state0 = os.path.join(station_corpus, "state0")
    if first_run and os.path.isdir(state0):
        for f in sorted(os.listdir(state0)):
            shutil.copy(os.path.join(state0, f), state_dir)
        print("Seeded lab state from state0: {:s}".format(
            ", ".join(sorted(os.listdir(state0)))))

    report = []
    prev_cal = None
    prev_model = None

    for night_name in night_names:

        src = os.path.join(station_corpus, night_name)
        dst = os.path.join(run_dir, "nights", night_name)
        os.makedirs(dst, exist_ok=True)

        # Inputs only - the archived products stay behind as the reference
        for pattern in INPUT_MEMBERS:
            for f in glob.glob(os.path.join(src, pattern)):
                shutil.copy(f, dst)

        # The station's own scoring product doubles as the forced-photometry
        # cache: the replay has no FF files, so the scoring pass sources the
        # forced channel from what the station measured (schema v5)
        ref_scoring = os.path.join(src, night_name + "_star_scoring.npz")
        if os.path.isfile(ref_scoring):
            from RMS.Formats.StarScoring import FORCED_CACHE_SUFFIX
            shutil.copy(ref_scoring, os.path.join(dst,
                "{:s}_{:s}".format(night_name, FORCED_CACHE_SUFFIX)))

        config = _labConfig(dst, station_id, repo_root)
        config.data_dir = state_dir

        print("\n=== {:s} ===".format(night_name))
        try:
            detectClouds(config, dst, show_plots=False, save_plots=False)
        except Exception as e:
            print("LAB WARNING: detectClouds raised: {!r}".format(e))

        # Make the night eligible for future auto-refits, like a real archive
        link = os.path.join(archived_dir, night_name)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(dst), link)

        cal = _calibrationState(state_dir, station_id)
        model = _modelState(state_dir, station_id)

        entry = dict(night=night_name, calibration=cal, model=model)

        # Lifecycle invariants
        if (prev_cal is not None) and (cal is not None):
            depth_changed = (prev_cal[2] is not None) and (cal[2] is not None) \
                and abs(prev_cal[2] - cal[2]) > 0.01
            if (cal[1] < prev_cal[1]) and not depth_changed:
                print("LAB WARNING: calibration regressed (max n_nights "
                    "{:d} -> {:d}) with NO depth change".format(prev_cal[1], cal[1]))
            if depth_changed:
                print("LAB NOTE: catalog depth changed {:.2f} -> {:.2f} - "
                    "calibration reset (max n_nights {:d} -> {:d})".format(
                    prev_cal[2], cal[2], prev_cal[1], cal[1] if cal else 0))
        if model is not None and model[2]:
            print("LAB WARNING: installed model carries quality issues: "
                "{}".format(model[2]))

        # Regression diff vs the station's own products
        for suffix in ("_transparency_map.npz", "_transparency_map_tree.npz"):
            produced = os.path.join(dst, night_name + suffix)
            reference = os.path.join(src, night_name + suffix)
            diff = _mapDiff(produced, reference)
            if diff is not None:
                entry["diff" + suffix.replace(".npz", "")] = diff
                print("map diff vs station {:s}: median |ddm| = {:.3f} mag".format(
                    suffix, diff))

        print("state: model={} calibration={}".format(model, cal))
        report.append(entry)
        prev_cal, prev_model = cal, model

    return report


def main():
    parser = argparse.ArgumentParser(description="Offline fleet replay lab "
        "for the transparency system.")
    parser.add_argument("corpus_dir", help="Corpus root (see module docstring)")
    parser.add_argument("--station", required=True, help="Station ID to replay")
    parser.add_argument("--lab", default=None, help="Lab run directory "
        "(default: <corpus>/_lab)")
    parser.add_argument("--fresh", action="store_true", help="Discard prior "
        "lab state for this station first")
    args = parser.parse_args()

    lab_dir = args.lab or os.path.join(args.corpus_dir, "_lab")
    report = replayStation(args.corpus_dir, args.station, lab_dir,
        fresh=args.fresh)

    print("\n==== replay summary ====")
    for e in report:
        print(json.dumps(e, default=str))


if __name__ == "__main__":
    main()
