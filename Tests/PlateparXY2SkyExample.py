"""Minimum viable example of mapping pixel coordinates to sky and ground coordinates using a platepar.

This script loads a platepar file, converts an image position (x, y) observed at a given
UTC time into J2000 right ascension/declination, precesses them to the epoch of date and
computes the corresponding alt/az with and without refraction. Finally, it projects the
no-refraction line of sight to a target height using the AEGeoidH2LatLonAlt helper.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import (
    AEGeoidH2LatLonAlt,
    J2000_JD,
    datetime2JD,
    raDec2AltAz,
    trueRaDec2ApparentAltAz,
)
from RMS.Astrometry.CyFunctions import equatorialCoordPrecession
from RMS.Formats.Platepar import Platepar


def parseTime(value: str) -> datetime:
    """Parse an ISO 8601 timestamp.

    The value is interpreted as UTC. Time zone aware values are converted to UTC and
    returned as naive ``datetime`` instances so they are compatible with RMS helpers.
    """

    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a pixel position from an RMS platepar into sky coordinates and"
            " ground intersection at a fixed height."
        )
    )
    parser.add_argument("platepar", help="Path to the platepar JSON/TXT file")
    parser.add_argument("x", type=float, help="Pixel column coordinate")
    parser.add_argument("y", type=float, help="Pixel row coordinate")
    parser.add_argument(
        "--time",
        dest="timestamp",
        required=True,
        type=parseTime,
        help="Observation time in ISO 8601 format (UTC). For example: 2023-01-15T12:34:56.789",
    )
    parser.add_argument(
        "--height",
        dest="height",
        type=float,
        default=10_000.0,
        help="Target height above mean sea level for the ground intercept (meters)",
    )

    args = parser.parse_args()

    platepar = Platepar()
    if platepar.read(args.platepar) is False:
        raise SystemExit(f"Could not read platepar file: {args.platepar}")

    # Convert the observation time to Julian Date using the station UT correction.
    jd = datetime2JD(args.timestamp, UT_corr=platepar.UT_corr)

    # Convert the pixel coordinates to J2000 RA/Dec using the platepar.
    # xyToRaDecPP expects iterables of equal length. Provide a single-sample input.
    _, ra_j2000, dec_j2000, _ = xyToRaDecPP(
        [jd],
        [args.x],
        [args.y],
        [1],
        platepar,
        jd_time=True,
        extinction_correction=False,
        precompute_pointing_corr=True,
    )

    ra_j2000 = float(ra_j2000[0])
    dec_j2000 = float(dec_j2000[0])

    # Manually precess the J2000 coordinates to the epoch of date.
    ra_epoch_rad, dec_epoch_rad = equatorialCoordPrecession(
        J2000_JD.days,
        jd,
        np.radians(ra_j2000),
        np.radians(dec_j2000),
    )
    ra_epoch = np.degrees(ra_epoch_rad)
    dec_epoch = np.degrees(dec_epoch_rad)

    # Convert to topocentric alt/az without refraction (relevant for in-atmosphere objects) using the 
    # epoch-of-date coordinates.
    az_no_ref, alt_no_ref = raDec2AltAz(ra_epoch, dec_epoch, jd, platepar.lat, platepar.lon)

    # Compute the apparent alt/az including refraction from the J2000 coordinates (relevant for 
    # exo-atmosphere objects).
    az_with_ref, alt_with_ref = trueRaDec2ApparentAltAz(
        ra_j2000,
        dec_j2000,
        jd,
        platepar.lat,
        platepar.lon,
        refraction=True,
    )

    # Project the non-refracted line of sight to a fixed height above the geoid.
    target_lat, target_lon = AEGeoidH2LatLonAlt(
        az_no_ref,
        alt_no_ref,
        args.height,
        platepar.lat,
        platepar.lon,
        platepar.elev,
    )

    print("Platepar:", args.platepar)
    print("Pixel coordinates:", f"x={args.x:.3f}", f"y={args.y:.3f}")
    print("Observation JD:", f"{jd:.8f}")
    print("RA/Dec (J2000):", f"ra={ra_j2000:.6f} deg", f"dec={dec_j2000:.6f} deg")
    print("RA/Dec (epoch of date):", f"ra={ra_epoch:.6f} deg", f"dec={dec_epoch:.6f} deg")
    print(
        "Alt/Az without refraction:",
        f"alt={alt_no_ref:.6f} deg",
        f"az={az_no_ref:.6f} deg",
    )
    print(
        "Alt/Az with refraction:",
        f"alt={alt_with_ref:.6f} deg",
        f"az={az_with_ref:.6f} deg",
    )
    print(
        "Ground intercept at",
        f"{args.height/1000.0:.2f} km:",
        f"lat={target_lat:.6f} deg",
        f"lon={target_lon:.6f} deg",
    )


if __name__ == "__main__":
    main()
