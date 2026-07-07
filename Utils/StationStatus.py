""" Headless station health report.

Prints a summary of how the last observing night went (using the observation summary database) plus
the current state of the station (disk space, camera reachability, next capture time). Designed to be
run over SSH on stations without a display:

    python -m Utils.StationStatus

or through the umbrella launcher:

    python -m RMS status
"""

from __future__ import print_function, division, absolute_import

import argparse
import datetime
import json
import os
import re
import shutil
import sys

from RMS.CLITools import addConfigArgument, loadConfig
from RMS.Formats.ObservationSummary import (OBSERVATION_DB_FILE_NAME, getObsDBConn,
    getLatestObservationRecord, getObservationDuration, pingOnce)


def getLastNightReport(config):
    """ Load the most recent night's record from the observation summary database.

    Return:
        [dict or None] Latest observation record, or None if no data is available yet.
    """

    db_path = os.path.join(config.data_dir, OBSERVATION_DB_FILE_NAME)

    # Don't create the database as a side effect of a read-only status check
    if not os.path.isfile(db_path):
        return None

    conn = getObsDBConn(config)

    if conn is None:
        return None

    try:
        record = getLatestObservationRecord(conn)
    finally:
        conn.close()

    return record


def getCurrentStatus(config):
    """ Gather the live station state - disk space, camera reachability, next capture time.

    Return:
        [dict] Current status values. Values which could not be determined are None.
    """

    status = {}

    # Free disk space in the data directory
    data_dir = os.path.expanduser(config.data_dir)
    try:
        usage = shutil.disk_usage(data_dir)
        status['disk_free_gb'] = usage.free/(1024**3)
        status['disk_total_gb'] = usage.total/(1024**3)
    except OSError:
        status['disk_free_gb'] = None
        status['disk_total_gb'] = None

    # Camera reachability - only for IP cameras where an IP can be extracted from the device string
    camera_ip_match = re.search(r'(?:\d{1,3}\.){3}\d{1,3}', str(config.deviceID))
    if camera_ip_match is not None:
        status['camera_ip'] = camera_ip_match.group()
        status['camera_reachable'] = pingOnce(status['camera_ip'])
    else:
        status['camera_ip'] = None
        status['camera_reachable'] = None

    # Next capture start and duration
    try:
        start_time, duration, end_time = getObservationDuration(config,
            datetime.datetime.now(datetime.timezone.utc))
        status['next_capture_start'] = str(start_time)
        status['next_capture_duration_hrs'] = duration/3600 if duration is not None else None
    except Exception:
        status['next_capture_start'] = None
        status['next_capture_duration_hrs'] = None

    return status


def formatReport(config, record, status):
    """ Format the status report as human-readable text. """

    def _get(key):
        if record is None:
            return 'n/a'
        value = record.get(key)
        return 'n/a' if value in (None, '') else str(value)

    lines = []
    lines.append("=" * 60)
    lines.append("RMS station status - {:s}".format(config.stationID))
    lines.append("=" * 60)

    if record is None:
        lines.append("")
        lines.append("No observation data yet - the station hasn't completed a night.")
        lines.append("")

    else:
        lines.append("")
        lines.append("Last night: {:s}".format(os.path.basename(_get('night_data_dir'))))
        lines.append("  FF files captured:      {:s} of {:s} expected".format(
            _get('total_fits'), _get('total_expected_fits')))
        lines.append("  Lost capture time:      {:s}".format(_get('fits_file_shortfall_as_time')))
        lines.append("  First / last FF:        {:s} / {:s}".format(
            _get('time_first_fits_file'), _get('time_last_fits_file')))
        lines.append("  First / last detection: {:s} / {:s}".format(
            _get('time_first_detection'), _get('time_last_detection')))
        lines.append("  Days since detection:   {:s}".format(_get('days_since_last_detection')))
        lines.append("  Tracebacks in log:      {:s}".format(_get('traceback_count')))
        lines.append("  Time sync:              {:s} (source: {:s}, offset {:s} ms)".format(
            _get('clock_synchronized'), _get('clock_measurement_source'), _get('clock_ahead_ms')))
        lines.append("  Repo lag behind remote: {:s} days".format(_get('repository_lag_remote_days')))
        lines.append("  Camera pointing:        az {:s}, alt {:s}, FOV {:s}x{:s} deg".format(
            _get('camera_pointing_az'), _get('camera_pointing_alt'),
            _get('camera_fov_h'), _get('camera_fov_v')))
        lines.append("")

    lines.append("Current state:")

    if status['disk_free_gb'] is not None:
        lines.append("  Disk free:              {:.1f} GB of {:.1f} GB ({:s})".format(
            status['disk_free_gb'], status['disk_total_gb'], config.data_dir))
    else:
        lines.append("  Disk free:              n/a (data directory not found)")

    if status['camera_ip'] is not None:
        lines.append("  Camera ({:s}):  {:s}".format(status['camera_ip'],
            "reachable" if status['camera_reachable'] else "NOT REACHABLE"))
    else:
        lines.append("  Camera:                 n/a (no IP camera configured)")

    if status['next_capture_start'] is not None:
        lines.append("  Next capture:           {:s} UTC, {:.1f} h".format(
            status['next_capture_start'], status['next_capture_duration_hrs']))
    else:
        lines.append("  Next capture:           n/a")

    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="""Print a headless station health report - how \
the last night went (capture completeness, detections, errors) and the current station state (disk, \
camera, next capture). Works over SSH, no display needed.""")

    addConfigArgument(arg_parser)

    arg_parser.add_argument('-j', '--json', action="store_true",
        help="Output the report as JSON for use in scripts.")

    cml_args = arg_parser.parse_args()

    config = loadConfig(cml_args.config)

    record = getLastNightReport(config)
    status = getCurrentStatus(config)

    if cml_args.json:
        print(json.dumps({'station_id': config.stationID, 'last_night': record, 'current': status},
            indent=4, sort_keys=True, default=str))

    else:
        print(formatReport(config, record, status))
