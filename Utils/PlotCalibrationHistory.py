""" Plot a station's long-term calibration tracking history.

The nightly pipeline maintains per-station tracking stores whose working values are
computed autonomously (trailing medians, step detectors, floor guards, seed handover -
see Utils.SkyQuality and the LM history in Utils.Flux). This tool renders that record
as one PNG per station, with every automated decision marked, so a glance answers
"did the automation do the right thing on the night that camera changed":

  <stationID>_radiometric.json      - nightly bias observations and floors (SkyQuality)
  <stationID>_flux_lm_history.json  - nightly matched depth and dome ratio (Flux)
  <night>/<night>_sky_quality.json  - nightly SQM records (optional, via --nights-dir)
  <night>/platepars_all_recalibrated.json - nightly photometric zero-point (optional)

The bias panel is a faithful REPLAY: the stored nights are folded through the same
resolveWorkingBias the pipeline runs, so the plotted working value, step detections,
floor guards and the seed handover are exactly what the pipeline decided (or would
have decided) each night.

Usage:
    python -m Utils.PlotCalibrationHistory <path to .config>
    python -m Utils.PlotCalibrationHistory <config> --nights-dir ~/RMS_data/ArchivedFiles
    python -m Utils.PlotCalibrationHistory --station US005X --data-dir ~/RMS_data
"""

from __future__ import absolute_import, division, print_function

import datetime
import json
import os

import numpy as np

from Utils.SkyQuality import RADIOMETRIC_FILE_SUFFIX, loadRadiometricCalibration, \
    resolveApertureCorrection, \
    resolveWorkingBias

# The LM history estimator constants live in Utils.Flux, which is heavy to import;
# fall back to their values if the import is unavailable in a stripped environment
try:
    from Utils.Flux import (LM_HISTORY_FILE, LM_HISTORY_WINDOW, LM_HISTORY_MIN_NIGHTS,
        LM_DEPTH_ENVELOPE_PCT, DOME_RATIO_NORM_PCT, DOME_RATIO_NORM_MIN,
        DOME_RATIO_NORM_MAX)
except Exception:
    LM_HISTORY_FILE = "flux_lm_history.json"
    LM_HISTORY_WINDOW = 30
    LM_HISTORY_MIN_NIGHTS = 5
    LM_DEPTH_ENVELOPE_PCT = 80
    DOME_RATIO_NORM_PCT = 80
    DOME_RATIO_NORM_MIN = 0.6
    DOME_RATIO_NORM_MAX = 2.5


def parseNightDate(key):
    """ Date of a night key: 'YYYYMMDD' or 'STATION_YYYYMMDD[_HHMMSS_us]'.

    Return:
        [datetime] or None if the key holds no date.
    """

    for part in str(key).split("_"):
        if len(part) == 8 and part.isdigit():
            try:
                return datetime.datetime.strptime(part, "%Y%m%d")
            except ValueError:
                return None

    return None


def replayBiasHistory(cal):
    """ Replay the nightly bias resolution over the stored history.

    Folds the stored nights, oldest first, through resolveWorkingBias with each night's
    own key, reproducing the working value and every step/guard/handover decision the
    pipeline made on that night.

    Arguments:
        cal: [dict] {seed, nights, ...} as loaded by loadRadiometricCalibration.

    Return:
        (records, handover): records is a list of dicts (date, obs, floor, bias, source)
            in date order; handover is the replayed handover record or None.
    """

    nights = {k: v for k, v in cal.get("nights", {}).items()
              if isinstance(v, dict) and (parseNightDate(k) is not None)}

    state = dict(seed=cal.get("seed"), nights={})
    ap_state = dict(seed=None, nights={})
    records = []

    for key in sorted(nights):
        entry = nights[key]
        if entry.get("floor") is None:
            continue

        bias, source, state = resolveWorkingBias(state, entry.get("bias"), entry["floor"],
            night_key=key)

        f_ap, _, ap_state = resolveApertureCorrection(ap_state, entry.get("aperture"),
            night_key=key)

        records.append(dict(date=parseNightDate(key), obs=entry.get("bias"),
            floor=entry["floor"], bias=bias, source=(source or ""),
            aperture_obs=entry.get("aperture"), aperture=f_ap))

    return records, state.get("handover")


def lmHistorySeries(history):
    """ Per-night depth/ratio series with the trailing estimators replayed per night.

    Arguments:
        history: [dict] night_key -> {depth, dratio, dmodel} as stored in the LM history.

    Return:
        series: [dict] with keys:
            depth: [(date, depth)], envelope: [(date, envelope)],
            ratio: [(date, dratio, version)], norm: [(date, norm)],
            version_changes: [(date, version)]
    """

    items = [(k, v) for k, v in sorted(history.items())
             if isinstance(v, dict) and (parseNightDate(k) is not None)]

    series = dict(depth=[], envelope=[], ratio=[], norm=[], version_changes=[])
    last_version = None

    for i, (key, entry) in enumerate(items):
        date = parseNightDate(key)

        # The estimators read the trailing window of PRIOR nights only
        prior = [v for _, v in items[max(0, i - LM_HISTORY_WINDOW):i]]

        if entry.get("depth") is not None:
            series["depth"].append((date, entry["depth"]))

            prior_depths = [v["depth"] for v in prior if v.get("depth") is not None]
            if len(prior_depths) >= LM_HISTORY_MIN_NIGHTS:
                series["envelope"].append(
                    (date, float(np.percentile(prior_depths, LM_DEPTH_ENVELOPE_PCT))))

        if entry.get("dratio") is not None:
            version = str(entry.get("dmodel", "unversioned"))
            series["ratio"].append((date, entry["dratio"], version))

            if version != last_version:
                if last_version is not None:
                    series["version_changes"].append((date, version))
                last_version = version

            prior_ratios = [v["dratio"] for v in prior if (v.get("dratio") is not None)
                            and (str(v.get("dmodel", "unversioned")) == version)]
            if len(prior_ratios) >= LM_HISTORY_MIN_NIGHTS:
                series["norm"].append((date, float(np.clip(
                    np.percentile(prior_ratios, DOME_RATIO_NORM_PCT),
                    DOME_RATIO_NORM_MIN, DOME_RATIO_NORM_MAX))))

    return series


def scanNightDirs(nights_dir, station_id):
    """ Collect nightly SQM and photometric zero-point records from archived night dirs.

    Arguments:
        nights_dir: [str] Directory holding night directories (e.g. ArchivedFiles).
        station_id: [str] Only directories starting with this station ID are read.

    Return:
        records: [list of dict] (date, sqm, absolute, mag_lev, mag_lev_spread) per night;
            keys are present only when the corresponding file was readable.
    """

    records = []

    for name in sorted(os.listdir(nights_dir)):
        night_dir = os.path.join(nights_dir, name)
        if not (name.startswith(str(station_id)) and os.path.isdir(night_dir)):
            continue

        date = parseNightDate(name)
        if date is None:
            continue

        rec = dict(date=date)

        for file_name in os.listdir(night_dir):
            if file_name.endswith("_sky_quality.json"):
                try:
                    with open(os.path.join(night_dir, file_name)) as f:
                        sq = json.load(f)
                    rec["sqm"] = sq.get("sqm")
                    rec["absolute"] = bool(sq.get("absolute"))
                except Exception:
                    pass

        pp_path = os.path.join(night_dir, "platepars_all_recalibrated.json")
        if os.path.isfile(pp_path):
            try:
                with open(pp_path) as f:
                    pps = json.load(f)
                mag_levs = [pp["mag_lev"] for pp in pps.values() if isinstance(pp, dict)
                            and pp.get("auto_recalibrated") and (pp.get("mag_lev") is not None)]
                if mag_levs:
                    rec["mag_lev"] = float(np.median(mag_levs))
                    rec["mag_lev_spread"] = float(np.std(mag_levs))
            except Exception:
                pass

        if len(rec) > 1:
            records.append(rec)

    return records


def plotCalibrationHistory(station_id, data_dir, nights_dir=None, output=None):
    """ Render the station's calibration history to a PNG.

    Arguments:
        station_id: [str]
        data_dir: [str] Directory holding the per-station tracking files.

    Keyword arguments:
        nights_dir: [str] Directory of archived night dirs; adds the SQM/zero-point panel.
        output: [str] Output PNG path. Default <data_dir>/<station>_calibration_history.png.

    Return:
        output: [str] The written PNG path, or None if there was nothing to plot.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = os.path.expanduser(data_dir)

    # --- gather ---

    class _Cfg(object):
        pass

    cfg = _Cfg()
    cfg.data_dir = data_dir
    cfg.stationID = str(station_id)

    bias_records, handover = [], None
    if os.path.isfile(os.path.join(data_dir,
            "{:s}_{:s}".format(str(station_id), RADIOMETRIC_FILE_SUFFIX))):
        cal = loadRadiometricCalibration(cfg)
        bias_records, handover = replayBiasHistory(cal)
        seed = cal.get("seed") or {}
    else:
        cal, seed = None, {}

    lm_series = None
    lm_path = os.path.join(data_dir, "{:s}_{:s}".format(str(station_id), LM_HISTORY_FILE))
    if os.path.isfile(lm_path):
        try:
            with open(lm_path) as f:
                lm_series = lmHistorySeries(json.load(f))
        except Exception:
            lm_series = None

    night_records = []
    if nights_dir and os.path.isdir(os.path.expanduser(nights_dir)):
        night_records = scanNightDirs(os.path.expanduser(nights_dir), station_id)

    panels = []
    if bias_records:
        panels.append("bias")
    if lm_series and lm_series["depth"]:
        panels.append("depth")
    if lm_series and lm_series["ratio"]:
        panels.append("ratio")
    if any(("sqm" in r) or ("mag_lev" in r) for r in night_records):
        panels.append("nights")
    if any(r.get("aperture_obs") is not None for r in bias_records):
        panels.append("aperture")

    if not panels:
        print("No calibration history found for {:s} in {:s}".format(
            str(station_id), data_dir))
        return None

    # --- render ---

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.8*len(panels)), sharex=True,
        squeeze=False)
    axes = axes.ravel()
    fig.suptitle("{:s} calibration history".format(str(station_id)))

    for ax, panel in zip(axes, panels):

        if panel == "bias":
            dates = [r["date"] for r in bias_records]
            obs = [(r["date"], r["obs"]) for r in bias_records if r["obs"] is not None]
            worked = [(r["date"], r["bias"]) for r in bias_records if r["bias"] is not None]

            ax.plot(dates, [r["floor"] for r in bias_records], color="0.75", lw=0.8,
                label="night floor")
            if obs:
                ax.plot(*zip(*obs), ls="none", marker=".", color="tab:blue",
                    label="nightly observation")
            if worked:
                ax.plot(*zip(*worked), drawstyle="steps-post", color="tab:blue", lw=1.6,
                    alpha=0.7, label="working bias")

            if seed.get("bias") is not None:
                ax.axhline(seed["bias"], ls="--", color="tab:green", lw=1.0,
                    label="seed ({:s})".format(str(seed.get("method", "stored"))))

            steps = [(r["date"], r["bias"]) for r in bias_records if "step" in r["source"]]
            guards = [r["date"] for r in bias_records if "floor guard" in r["source"]]
            if steps:
                ax.plot(*zip(*steps), ls="none", marker="v", color="tab:red", ms=8,
                    label="pedestal step")
            for i, d in enumerate(guards):
                ax.axvline(d, color="tab:orange", lw=1.0, alpha=0.6,
                    label=("floor guard" if i == 0 else None))

            if handover:
                d = parseNightDate(handover.get("date"))
                if d is not None:
                    ax.axvline(d, color="tab:green", lw=1.2, alpha=0.8)
                    ax.annotate("seed handover ({:+.1f} ADU)".format(handover["delta"]),
                        xy=(d, 0.95), xycoords=("data", "axes fraction"),
                        fontsize=8, color="tab:green", rotation=90, va="top", ha="right")

            ax.set_ylabel("bias (ADU)")
            ax.legend(loc="best", fontsize=8, ncol=3)

        elif panel == "aperture":
            ap_obs = [(r["date"], r["aperture_obs"]) for r in bias_records
                      if r.get("aperture_obs") is not None]
            ap_work = [(r["date"], r["aperture"]) for r in bias_records
                       if r.get("aperture") is not None]
            if ap_obs:
                ax.plot(*zip(*ap_obs), ls="none", marker=".", color="tab:green",
                    label="nightly capture fraction f")
            if ap_work:
                ax.plot(*zip(*ap_work), drawstyle="steps-post", color="tab:green",
                    lw=1.6, alpha=0.7, label="working f (trailing median)")
            ax.set_ylabel("PSF capture fraction")
            ax.set_ylim(0.3, 1.2)
            ax.legend(loc="upper left", fontsize=8)

        elif panel == "depth":
            ax.plot(*zip(*lm_series["depth"]), ls="none", marker=".", color="tab:purple",
                label="night matched depth")
            if lm_series["envelope"]:
                ax.plot(*zip(*lm_series["envelope"]), color="tab:purple", lw=1.6,
                    alpha=0.7, label="clear-sky envelope (P{:d}/{:d} nights)".format(
                        LM_DEPTH_ENVELOPE_PCT, LM_HISTORY_WINDOW))
            ax.set_ylabel("depth (mag)")
            ax.legend(loc="best", fontsize=8)

        elif panel == "ratio":
            versions = []
            for _, _, v in lm_series["ratio"]:
                if v not in versions:
                    versions.append(v)
            for i, version in enumerate(versions):
                pts = [(d, r) for d, r, v in lm_series["ratio"] if v == version]
                ax.plot(*zip(*pts), ls="none", marker=".",
                    color="C{:d}".format(i % 10), label="ratio (model {:s})".format(version))
            if lm_series["norm"]:
                ax.plot(*zip(*lm_series["norm"]), color="0.3", lw=1.4, alpha=0.8,
                    label="normalization (P{:d}, clipped)".format(DOME_RATIO_NORM_PCT))
            for d, version in lm_series["version_changes"]:
                ax.axvline(d, ls="--", color="0.5", lw=1.0)
            ax.axhline(1.0, color="0.8", lw=0.8)
            ax.set_ylabel("dome ratio")
            ax.legend(loc="best", fontsize=8, ncol=2)

        elif panel == "nights":
            sqm_abs = [(r["date"], r["sqm"]) for r in night_records
                       if r.get("sqm") is not None and r.get("absolute")]
            sqm_lim = [(r["date"], r["sqm"]) for r in night_records
                       if r.get("sqm") is not None and not r.get("absolute")]
            if sqm_abs:
                ax.plot(*zip(*sqm_abs), ls="none", marker=".", color="tab:cyan",
                    label="SQM")
            if sqm_lim:
                # Limits bound the sky from below: at least this bright
                ax.plot(*zip(*sqm_lim), ls="none", marker="^", mfc="none",
                    color="tab:cyan", label="SQM limit")
            ax.set_ylabel("SQM (mag/arcsec$^2$)")
            ax.invert_yaxis()

            mag = [(r["date"], r["mag_lev"], r.get("mag_lev_spread", 0.0))
                   for r in night_records if r.get("mag_lev") is not None]
            if mag:
                ax2 = ax.twinx()
                md, mv, ms = zip(*mag)
                ax2.errorbar(md, mv, yerr=ms, ls="none", marker=".", color="tab:brown",
                    elinewidth=0.6, ms=4, alpha=0.8, label="zero-point (mag_lev)")
                ax2.set_ylabel("zero-point (mag)", color="tab:brown")
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
            else:
                ax.legend(loc="best", fontsize=8)

        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("night")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output is None:
        output = os.path.join(data_dir,
            "{:s}_calibration_history.png".format(str(station_id)))

    fig.savefig(output, dpi=120)
    plt.close(fig)

    return output


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Plot a station's long-term calibration tracking history.")
    arg_parser.add_argument("config_path", nargs="?",
        help="Path to the station's .config file (or use --station with --data-dir).")
    arg_parser.add_argument("--station", help="Station ID (overrides the config).")
    arg_parser.add_argument("--data-dir",
        help="RMS data directory holding the tracking files (overrides the config).")
    arg_parser.add_argument("--nights-dir",
        help="Directory of archived night directories - adds the SQM/zero-point panel.")
    arg_parser.add_argument("-o", "--output", help="Output PNG path.")

    cml_args = arg_parser.parse_args()

    station_id, data_dir = cml_args.station, cml_args.data_dir

    if cml_args.config_path:
        import RMS.ConfigReader as cr
        config = cr.parse(os.path.expanduser(cml_args.config_path))
        station_id = station_id or config.stationID
        data_dir = data_dir or config.data_dir

    if not (station_id and data_dir):
        arg_parser.error("give a config file, or both --station and --data-dir")

    out = plotCalibrationHistory(station_id, data_dir, nights_dir=cml_args.nights_dir,
        output=cml_args.output)

    if out:
        print("Wrote {:s}".format(out))
