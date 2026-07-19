""" Build an integrated-starlight (ISL) sky brightness map from the star catalog
(EXPERIMENTAL - not yet wired into the light-dome model).

The Milky Way's diffuse glow is a celestial-fixed sky-brightness component that the
light-dome model's ground-fixed terms (airglow + light pollution) cannot represent. At
dark stations the in-plane glow is a significant fraction of the total background
(in-plane surface brightness ~21-22 mag/arcsec^2 vs a ~21.8 pristine sky), so the local
limiting magnitude genuinely dips as the plane transits - unmodeled today.

This tool sums catalog starlight on a galactic-coordinate grid and writes a compact map
(mag/arcsec^2 per cell) for later use as an additive flux term in the dome model's
brightness composition, following the standard decomposition of Leinert et al. (1998):
zodiacal light + integrated starlight + diffuse galactic light + airglow.

LIMITATIONS (why this is a prototype):
- The shipped catalogs reach mag ~11.5-12; integrated starlight receives roughly
  comparable contributions per magnitude out to mag ~18, so the direct sum captures
  only part of the total. ISL_COMPLETION_FACTOR scales the summed flux to approximate
  the full column (literature-guided, and it should eventually be FIT against dark-site
  sky measurements rather than assumed).
- Diffuse galactic light (starlight scattered by dust) adds ~20-30% on top of ISL in
  the plane; folded into the same factor.
- Zodiacal light is NOT included here (it needs ecliptic geometry and solar elongation
  at evaluation time, not a static map).

Usage:
    python -m Utils.BuildISLMap /path/to/station_config_dir [--out isl_map.npz]
"""

from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np

# Grid resolution (degrees). The MW structure relevant at a 10-arcmin beam and a
# 45+ deg FOV is broad; 1 x 1 deg is plenty.
GRID_STEP_DEG = 1.0

# Scale from summed catalog flux (to the catalog limit) to the full ISL+DGL column.
# Order-of-magnitude from the Leinert et al. (1998) tables for a mag ~12 cutoff;
# marked for empirical fitting against dark-site sky data.
ISL_COMPLETION_FACTOR = 2.2

# Galactic north pole (J2000)
NGP_RA, NGP_DEC = 192.8595, 27.1284
GAL_L_NCP = 122.932


def equatorialToGalactic(ra_deg, dec_deg):
    """ J2000 equatorial to galactic coordinates (degrees, vectorized). """

    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    ngp_ra, ngp_dec = np.radians(NGP_RA), np.radians(NGP_DEC)

    sb = (np.sin(dec)*np.sin(ngp_dec)
          + np.cos(dec)*np.cos(ngp_dec)*np.cos(ra - ngp_ra))
    b = np.arcsin(np.clip(sb, -1.0, 1.0))

    y = np.cos(dec)*np.sin(ra - ngp_ra)
    x = (np.sin(dec)*np.cos(ngp_dec)
         - np.cos(dec)*np.sin(ngp_dec)*np.cos(ra - ngp_ra))
    l = np.radians(GAL_L_NCP) - np.arctan2(y, x)

    return np.degrees(l)%360.0, np.degrees(b)


def buildISLMap(catalog_stars, grid_step=GRID_STEP_DEG,
                completion_factor=ISL_COMPLETION_FACTOR):
    """ Sum catalog starlight per galactic-coordinate cell.

    Arguments:
        catalog_stars: [ndarray] (N, 3) of (RA, Dec, Mag).

    Keyword arguments:
        grid_step: [float] Cell size in degrees.
        completion_factor: [float] Flux scale to the full ISL+DGL column.

    Return:
        map_dict: [dict]
            l_edges, b_edges: [ndarray] Cell edges (deg).
            sb: [ndarray] Surface brightness per cell, mag/arcsec^2 (inf where empty).
    """

    ra, dec, mag = catalog_stars.T
    l, b = equatorialToGalactic(ra, dec)
    flux = 10.0**(-0.4*mag)

    l_edges = np.arange(0.0, 360.0 + grid_step/2, grid_step)
    b_edges = np.arange(-90.0, 90.0 + grid_step/2, grid_step)

    flux_sum, _, _ = np.histogram2d(l, b, bins=[l_edges, b_edges], weights=flux)

    # Cell solid angle in arcsec^2 (depends on b)
    b_centers = (b_edges[:-1] + b_edges[1:])/2
    cell_arcsec2 = (grid_step*3600.0)**2*np.cos(np.radians(b_centers))
    cell_arcsec2 = np.maximum(cell_arcsec2, 1.0)

    with np.errstate(divide="ignore"):
        sb = -2.5*np.log10(completion_factor*flux_sum/cell_arcsec2[None, :])

    return dict(l_edges=l_edges, b_edges=b_edges, sb=sb,
                grid_step=grid_step, completion_factor=completion_factor)


def evalISL(map_dict, l_deg, b_deg):
    """ Surface brightness (mag/arcsec^2) at the given galactic coordinates. """

    li = np.clip(((np.asarray(l_deg)%360.0)/map_dict["grid_step"]).astype(int),
                 0, map_dict["sb"].shape[0] - 1)
    bi = np.clip(((np.asarray(b_deg) + 90.0)/map_dict["grid_step"]).astype(int),
                 0, map_dict["sb"].shape[1] - 1)

    return map_dict["sb"][li, bi]


if __name__ == "__main__":

    import RMS.ConfigReader as cr
    from RMS.Formats import StarCatalog

    parser = argparse.ArgumentParser(description="Build an ISL map from the star catalog")
    parser.add_argument("config_dir", help="Station config directory")
    parser.add_argument("--out", default="isl_map.npz")
    parser.add_argument("--lim-mag", type=float, default=12.0)
    args = parser.parse_args()

    config = cr.loadConfigFromDirectory(".", args.config_dir)
    cat, _, _ = StarCatalog.readStarCatalog(config.star_catalog_path,
        config.star_catalog_file, lim_mag=args.lim_mag,
        mag_band_ratios=config.star_catalog_band_ratios)

    m = buildISLMap(cat)
    np.savez_compressed(args.out, **m)

    lc = (m["l_edges"][:-1] + m["l_edges"][1:])/2
    bc = (m["b_edges"][:-1] + m["b_edges"][1:])/2
    plane = m["sb"][:, np.abs(bc) < 5].ravel()
    pole = m["sb"][:, np.abs(bc) > 60].ravel()
    print("ISL map written to {:s}".format(args.out))
    print("galactic plane |b|<5:  median {:.2f} mag/arcsec^2".format(
        np.median(plane[np.isfinite(plane)])))
    print("galactic poles |b|>60: median {:.2f} mag/arcsec^2".format(
        np.median(pole[np.isfinite(pole)])))
