#!/usr/bin/env python3
"""
deadman_camera_watchdog.py

Hourly watchdog that keeps a network camera's auto-reboot timer pushed
one hour beyond the next top-of-hour.

* Startup  - probe once; if OK, SetAutoReboot "Everyday,<next-hour>".
* Hourly   - wake up a little before HH:00, retry-probe, and reset the
             timer to "<next-hour+1>" so the reboot fires **only** if
             the watchdog fails for a full hour.

All times are logged and applied in UTC.
"""

from __future__ import print_function, division, absolute_import

import argparse
import datetime as _dt
import errno
import os
import re
import socket
import time
from urllib.parse import urlparse

import Utils.CameraControl as cc
from RMS.Logger import getLogger

# ──────────── constants ────────────
RETRIES        = 5
RETRY_DELAY    = 10
LEAD_SECONDS   = RETRIES * RETRY_DELAY + 20   # wake-up margin before HH:00

log = getLogger("deadman")

# ──────────── helpers ────────────
class RtspProbeResult:
    SUCCESS, NETWORK_DOWN, HOST_UNREACHABLE, CONNECTION_REFUSED, \
    TIMEOUT, DNS_ERROR, UNKNOWN_ERROR = range(7)

def extractRtspUrl(text):
    m = re.search(r"rtsp://[^\s]+", text)
    if not m:
        raise ValueError(f"No RTSP URL in: {text!r}")
    return m.group(0)

def isRtspUp(rtsp_url, timeout=1):
    host, port = urlparse(rtsp_url).hostname, urlparse(rtsp_url).port or 554
    try:
        socket.gethostbyname(host)
        with socket.create_connection((host, port), timeout):
            return True, RtspProbeResult.SUCCESS
    except socket.gaierror:
        return False, RtspProbeResult.DNS_ERROR
    except socket.timeout:
        return False, RtspProbeResult.TIMEOUT
    except OSError as e:
        mapping = {errno.ENETUNREACH:  RtspProbeResult.NETWORK_DOWN,
                   errno.EHOSTUNREACH: RtspProbeResult.HOST_UNREACHABLE,
                   errno.ECONNREFUSED: RtspProbeResult.CONNECTION_REFUSED}
        return False, mapping.get(e.errno, RtspProbeResult.UNKNOWN_ERROR)

def probeWithRetry(rtsp_url, retries=RETRIES, delay=RETRY_DELAY):
    for n in range(1, retries + 1):
        ok, _ = isRtspUp(rtsp_url)
        if ok:
            log.debug(f"RTSP OK on attempt {n}")
            return True
        time.sleep(delay)
    return False

def nextHourUtc():
    """Return timezone-aware datetime for the next top-of-hour (UTC)."""
    now = _dt.datetime.now(_dt.timezone.utc)
    return (now.replace(minute=0, second=0, microsecond=0)
              + _dt.timedelta(hours=1))

# ──────────── core loop ────────────
def deadmanLoop(config,
                retries=RETRIES,
                retry_delay=RETRY_DELAY,
                lead_seconds=LEAD_SECONDS):

    rtsp_url = extractRtspUrl(config.deviceID)

    # ── startup probe ────────────────────────────────────────────────
    now   = _dt.datetime.now(_dt.timezone.utc)
    ok, _ = isRtspUp(rtsp_url)
    if ok:
        reboot_hr = nextHourUtc().hour                # fire at next top-of-hour
        cc.cameraControlV2(config, "SetAutoReboot",
                           [f"Everyday,{reboot_hr}"])
        log.info(f"[{now:%Y-%m-%d %H:%M:%S}Z] startup probe OK -> "
                 f"auto-reboot set for {reboot_hr:02d}:00Z")
    else:
        log.warning(f"[{now:%Y-%m-%d %H:%M:%S}Z] startup probe FAILED")

    # ── hourly loop ─────────────────────────────────────────────────
    while True:
        now       = _dt.datetime.now(_dt.timezone.utc)
        next_top  = nextHourUtc()
        time.sleep(max(0, (next_top - now).total_seconds() - lead_seconds))

        # retry-aware probe just before HH:00
        if probeWithRetry(rtsp_url, retries=retries, delay=retry_delay):
            reboot_hr = (next_top.hour + 1) % 24      # push one hour further
            cc.cameraControlV2(config, "SetAutoReboot",
                               [f"Everyday,{reboot_hr}"])
            log.info(f"[{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d %H:%M:%S}Z] "
                     f"camera healthy → auto-reboot bumped to {reboot_hr:02d}:00Z")
        else:
            log.warning(f"[{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d %H:%M:%S}Z] "
                        "camera offline after retries; leaving previous schedule")

        time.sleep(lead_seconds)   # idle until HH:00 actually flips


# ─────────────────────────────── CLI ────────────────────────────────────
if __name__ == "__main__":
    import RMS.ConfigReader as cr

    # ---- command-line args ----
    parser = argparse.ArgumentParser(description="Dead-man watchdog for RTSP cameras.")
    parser.add_argument(
        "-c", "--config",
        nargs=1,
        metavar="CONFIG_PATH",
        type=str,
        help="Directory containing the RMS config file (default: current directory)."
    )
    cml_args = parser.parse_args()
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    deadmanLoop(config)