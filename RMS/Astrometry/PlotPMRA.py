#!/usr/bin/env python3
"""Plot pmra vs declination from a GMN star catalog binary file.

The scatter plot helps diagnose whether PMRA has already been
multiplied by 1/cos(dec) ("true" pmra) or is still Gaia’s compressed
μ_α* (pmra·cos δ). If the latter, points should collapse toward
zero as |dec|→90°.

Usage:
    python plot_pmra_vs_dec.py [GMN_StarCatalog.bin] [--sample 100000] [--save fig.png]
        # Default catalog:
        # /Users/lucbusquin/Projects/RMS/Catalogs/GMN_StarCatalog_LM9.0.bin
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path

# Record layout used when the GMN catalog was built
# (matches the `data_types` list defined in GMN scripts)
CAT_DTYPE = np.dtype([
    ("designation", "S30"),
    ("ra", "f8"),        # degrees, epoch J2000
    ("dec", "f8"),       # degrees, epoch J2000
    ("pmra", "f8"),      # mas/yr or μ_α* (pmra·cos δ)
    ("pmdec", "f8"),
    ("phot_g_mean_mag", "f4"),
    ("phot_bp_mean_mag", "f4"),
    ("phot_rp_mean_mag", "f4"),
    ("classprob_dsc_specmod_star", "f4"),
    ("classprob_dsc_specmod_binarystar", "f4"),
    ("spectraltype_esphs", "S8"),
    ("B", "f4"),
    ("V", "f4"),
    ("R", "f4"),
    ("Ic", "f4"),
    ("oid", "i4"),
    ("preferred_name", "S30"),
    ("Simbad_OType", "S30"),
])


def load_catalog(path: Path, sample: Optional[int] = None) -> np.ndarray:
    """
    Load a GMN star‑catalog binary (.bin) file.

    The file layout is:
        uint32  declared_header_size
        uint32  num_rows
        uint32  num_columns
        bytes   column‑name table  (declared_header_size‑12 bytes)
        bytes   zlib‑compressed structured records
    """
    import zlib

    with path.open("rb") as f:
        header_size  = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        num_rows     = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        _num_cols    = int(np.fromfile(f, dtype=np.uint32, count=1)[0])

        # skip the column‑name block
        f.read(header_size - 12)

        # read & decompress the payload
        compressed    = f.read()
        decompressed  = zlib.decompress(compressed)

    data = np.frombuffer(decompressed, dtype=CAT_DTYPE, count=num_rows)

    # Optional random down‑sample to speed up plotting
    if sample and sample < len(data):
        rng  = np.random.default_rng(0)
        idx  = rng.choice(len(data), size=sample, replace=False)
        data = data[idx]

    return data


def plot_pmra_vs_dec(catalog: np.ndarray, save: Optional[str] = None):
    dec = catalog["dec"]
    pmra = catalog["pmra"]

    # keep only sane data
    mask = (
        np.isfinite(dec) & np.isfinite(pmra) &
        (dec > -90) & (dec < 90) &
        (np.abs(pmra) < 1e5)           # guard against crazy outliers
    )
    dec = dec[mask]
    pmra = pmra[mask]

    plt.figure(figsize=(7, 4))
    plt.scatter(pmra, dec, s=1, alpha=0.3)
    plt.xlabel("pmra (mas yr$^{-1}$)")
    plt.ylabel("Declination (deg)")
    plt.title("GMN star catalog: Dec vs pmra")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot pmra vs dec from a GMN star catalog binary file.")
    parser.add_argument(
        "catalog",
        nargs="?",
        default="/Users/lucbusquin/Projects/RMS/Catalogs/GMN_StarCatalog_LM9.0.bin",
        help="Path to GMN_StarCatalog.bin (default: %(default)s)",
    )
    parser.add_argument("--sample", type=int, default=None, help="Randomly subsample N stars for faster plotting")
    parser.add_argument("--save", metavar="PNG", help="Save figure instead of showing interactively")
    args = parser.parse_args()

    catalog = load_catalog(Path(args.catalog), args.sample)
    plot_pmra_vs_dec(catalog, args.save)


if __name__ == "__main__":
    main()
