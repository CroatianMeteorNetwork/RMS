""" Long-term sky quality plot for a single station.

One universal design that works for any station - single camera, multi-camera, zenith
pointing or not: the station's nightly measured SQM (fixed FOV patch) is normalized to a
ZENITH-EQUIVALENT value using the station's own light-dome model, which puts every
station on the same absolute scale (Bortle bands apply) while preserving the station's
trend exactly (a fixed patch normalizes by a constant).

Measured values are stored; the normalization is applied at PLOT time with the current
model, so a model refit re-normalizes the whole series consistently instead of stepping
it. Without a model (fresh station warmup) the measured series is plotted unnormalized
and labeled accordingly - same code path, degraded gracefully.

The plot is written to <stationID>_sky_quality.png in the data directory and copied into
the night directory (archive) by the caller.
"""

from __future__ import absolute_import, division, print_function

import datetime
import json
import os
import re

import numpy as np

from RMS.Logger import getLogger

log = getLogger("logger")


SKY_HISTORY_SUFFIX = "sky_quality_history.json"
SKY_HISTORY_KEEP = 3660          # ~10 years of nightly records
NEAR_ZENITH_ALT = 70.0           # deg - above this the normalization is negligible
UNCERT_NEAR_ZENITH = 0.1         # mag - stated normalization uncertainty near zenith
UNCERT_LOW_ALT = 0.5             # mag - stated normalization uncertainty for low patches

# Approximate zenith SQM to Bortle class band edges (mag/arcsec^2)
BORTLE_BANDS = [(21.99, 21.69, "3"), (21.69, 21.25, "4"), (21.25, 20.49, "5"),
                (20.49, 19.50, "6"), (19.50, 18.94, "7"), (18.94, 18.38, "8"),
                (18.38, 17.50, "9")]


def _historyPath(config):
    return os.path.join(os.path.expanduser(config.data_dir),
        "{:s}_{:s}".format(str(config.stationID), SKY_HISTORY_SUFFIX))


def _nightDateKey(night_name):
    """ YYYYMMDD date chunk of a night directory name, located by pattern (works for any
        stationID scheme). """

    match = re.search(r"_(\d{8})(?:_|$)", night_name)

    return match.group(1) if match else night_name


def appendSkyQualityHistory(config, night_name, record):
    """ Append one night's sky quality outcome to the station's long-term history.

    Measured nights store the MEASURED sqm and the patch position; normalization happens
    at plot time. Skipped nights store their status so coverage is auditable.

    Arguments:
        config: [Config]
        night_name: [str] Night directory name.
        record: [dict] The sky quality record as written to the night directory.
    """

    entry = dict(status=record.get("status", "ok"))

    if record.get("sqm") is not None:
        entry.update(sqm=record["sqm"], absolute=bool(record.get("absolute")),
            az=record.get("patch", {}).get("az"), alt=record.get("patch", {}).get("alt"),
            tier=record.get("bias", {}).get("tier"))
    else:
        entry["reason"] = record.get("reason")

    path = _historyPath(config)
    history = {}
    try:
        if os.path.isfile(path):
            with open(path) as f:
                history = json.load(f)
    except Exception:
        history = {}

    nights = history.get("nights", {})
    nights[_nightDateKey(night_name)] = entry
    history["nights"] = dict(sorted(nights.items())[-SKY_HISTORY_KEEP:])

    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=1)
    except Exception as e:
        log.debug("Could not write the sky quality history: {}".format(e))


def zenithDelta(dome_model, az, alt):
    """ Magnitude offset from the patch position to the zenith, per the model. """

    if (dome_model is None) or (az is None) or (alt is None):
        return None

    b_patch = float(dome_model.skyBrightness(float(az), float(alt)))
    b_zenith = float(dome_model.skyBrightness(0.0, 90.0))

    return 2.5*np.log10(b_patch/b_zenith)


def plotStationSkyQuality(config, dome_model=None):
    """ Render the station's long-term sky quality plot from its history.

    Arguments:
        config: [Config]

    Keyword arguments:
        dome_model: [LightDomeModel] Used for zenith normalization; None plots the
            measured series unnormalized (graceful degradation, labeled).

    Return:
        out_path: [str] Path of the written PNG, or None if there is nothing to plot.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    path = _historyPath(config)
    if not os.path.isfile(path):
        return None

    try:
        with open(path) as f:
            nights = json.load(f).get("nights", {})
    except Exception:
        return None

    if not nights:
        return None

    days, values, statuses, limits = [], [], [], []
    alts = []
    normalized = dome_model is not None

    for key in sorted(nights):
        try:
            day = datetime.datetime.strptime(key, "%Y%m%d")
        except ValueError:
            continue

        entry = nights[key]
        days.append(day)
        statuses.append(entry.get("status", "ok") if entry.get("sqm") is not None
                        or entry.get("status") != "ok" else "ok")

        if entry.get("sqm") is not None:
            value = float(entry["sqm"])
            delta = zenithDelta(dome_model, entry.get("az"), entry.get("alt"))
            if delta is not None:
                value += delta
            alts.append(entry.get("alt"))
            if entry.get("absolute", True):
                values.append(value)
                limits.append(None)
            else:
                values.append(None)
                limits.append(value)
        else:
            values.append(None)
            limits.append(None)

    if not any(v is not None for v in values) and not any(v is not None for v in limits):
        return None

    fig, (ax, ax_status) = plt.subplots(2, 1, figsize=(11.5, 5.8), sharex=True,
        gridspec_kw=dict(height_ratios=[6, 0.5], hspace=0.05))

    # Bortle bands only make sense on the zenith scale
    if normalized:
        for top, bot, name in BORTLE_BANDS:
            ax.axhspan(bot, top, color=plt.cm.inferno_r(0.12 + 0.11*int(name)),
                alpha=0.18, zorder=0)

    xs = [d for d, v in zip(days, values) if v is not None]
    ys = [v for v in values if v is not None]
    if xs:
        ax.plot(xs, ys, "o-", color="#202020", ms=6, lw=1.1, label="nightly SQM")

        if len(ys) >= 5:
            trend = [float(np.median(ys[max(0, i - 6):i + 1])) for i in range(len(ys))]
            ax.plot(xs, trend, "--", color="#c04040", lw=1.1, alpha=0.8,
                label="7-night median")

    # Limits: sky at least this bright (bias unknown that night)
    lx = [d for d, v in zip(days, limits) if v is not None]
    ly = [v for v in limits if v is not None]
    if lx:
        ax.plot(lx, ly, "v", color="#888", ms=7, fillstyle="none", label="limit (no bias)")

    ax.invert_yaxis()
    median_alt = float(np.median([a for a in alts if a is not None])) if alts else None

    if normalized and median_alt is not None:
        uncert = UNCERT_NEAR_ZENITH if median_alt >= NEAR_ZENITH_ALT else UNCERT_LOW_ALT
        subtitle = ("zenith-normalized from the fixed patch at alt {:.0f} deg via the "
            "station light-dome model (normalization uncertainty ~{:.1f} mag)").format(
            median_alt, uncert)
        ax.set_ylabel("zenith-equivalent SQM (mag/arcsec$^2$)")
    else:
        subtitle = "measured at the fixed patch - zenith normalization pending a sky model"
        ax.set_ylabel("measured SQM (mag/arcsec$^2$)")

    ax.set_title("{:s} nightly sky quality\n{:s}".format(str(config.stationID), subtitle),
        fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")

    # A properly labeled right-hand scale for the Bortle bands
    if normalized:
        ax_bortle = ax.twinx()
        ax_bortle.set_ylim(ax.get_ylim())
        ax_bortle.set_yticks([(top + bot)/2.0 for top, bot, _ in BORTLE_BANDS])
        ax_bortle.set_yticklabels([name for _, _, name in BORTLE_BANDS], fontsize=8)
        ax_bortle.set_ylabel("Bortle class (approximate)", fontsize=9)
        ax_bortle.tick_params(length=0)

    # Status strip
    colors = {"ok": "#3a3", "skipped": "#bbb"}
    for day, entry_key in zip(days, sorted(nights)):
        st = nights[entry_key].get("status", "ok")
        x = mdates.date2num(day)
        ax_status.barh(0, 0.8, left=x - 0.4, height=0.8,
            color=colors.get(st, "#bbb"), edgecolor="gray", linewidth=0.3)
    ax_status.set_yticks([])
    ax_status.set_ylim(-0.6, 0.6)
    ax_status.set_xlabel("night (green = measured, grey = skipped/cloudy)", fontsize=8)
    ax_status.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    out_path = os.path.join(os.path.expanduser(config.data_dir),
        "{:s}_sky_quality.png".format(str(config.stationID)))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return out_path


if __name__ == "__main__":

    import argparse
    import RMS.ConfigReader as cr
    from RMS.LightDomeModel import LightDomeModel

    parser = argparse.ArgumentParser(
        description="Render a station's long-term sky quality plot from its history.")
    parser.add_argument("station_dir", help="Station config directory.")

    args = parser.parse_args()

    config = cr.loadConfigFromDirectory(".", os.path.abspath(args.station_dir))
    out = plotStationSkyQuality(config, dome_model=LightDomeModel.load(config))
    print(out if out else "nothing to plot")
