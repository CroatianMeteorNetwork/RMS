""" Make a DXF outline of a physical light occluder from a mask image and a platepar.

The masked regions of the image (value 0, e.g. street lights) are mapped through the platepar to a plane
perpendicular to the optical axis at a given distance in front of the lens. The DXF holds the shape an
opaque plate in that plane must have to block them. All lengths are in mm.

A pixel's chief ray leaves the entrance pupil in the direction (tx, ty) and, at the plane, the beam of
that pixel is a disk of radius pupil/2 centred at D*(tx, ty). With M the masked region in tangent
coordinates, F the frame border and r = pupil/2 + margin, the DXF layers are:

    CLEAR           D*F grown by r      what the camera sees through at the plane
    OCCLUDER_FULL   D*M grown by r      blocks every masked direction completely (kept pixels next to the
                                        mask edge are partly vignetted)
    OCCLUDER_SAFE   D*M shrunk by r     never touches the beam of a kept pixel (masked pixels next to the
                                        mask edge are only partly blocked, thin features vanish)
    OPENING_FULL    CLEAR - OCCLUDER_FULL
    OPENING_SAFE    CLEAR - OCCLUDER_SAFE
    AXIS            a 10 mm cross on the optical axis

Coordinates: x = image +x, y = image +y (down), origin on the optical axis, as seen from behind the
camera looking out.

Needs the shapely and ezdxf packages (pip install shapely ezdxf).
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import sys
import logging
import argparse

import numpy as np
import cv2

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz
from RMS.Astrometry.CyFunctions import refractionScale


DXF_LAYERS = ["CLEAR", "OCCLUDER_FULL", "OCCLUDER_SAFE", "OPENING_FULL", "OPENING_SAFE"]


def _importGeometryPackages():
    """ Import shapely and ezdxf when they are needed, so that RMS does not depend on them. """

    try:
        import shapely
        import shapely.ops
        import shapely.affinity
        import shapely.geometry
        import ezdxf

    except ImportError as e:
        raise ImportError("The occluder DXF export needs the shapely and ezdxf packages ({:s}). "
            "Install them with: pip install shapely ezdxf".format(str(e)))

    # ezdxf logs every table it creates at the INFO level
    logging.getLogger("ezdxf").setLevel(logging.WARNING)

    return shapely, ezdxf


def pixelsToDirections(platepar, x, y, jd):
    """ Convert image pixels to unit vectors in the local (East, North, Up) frame, as the camera sees
        them (apparent directions at the given Julian date).

    Arguments:
        platepar: [Platepar object]
        x: [array like] Image X coordinates (px).
        y: [array like] Image Y coordinates (px).
        jd: [float] Julian date.

    Return:
        v: [ndarray (N, 3)] Unit vectors.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    _, ra, dec, _ = xyToRaDecPP([float(jd)]*n, x, y, [1]*n, platepar, extinction_correction=False,
        jd_time=True, precompute_pointing_corr=True)

    azim, alt = trueRaDec2ApparentAltAz(np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64),
        float(jd), float(platepar.lat), float(platepar.lon), refraction=bool(platepar.refraction),
        refraction_scale=refractionScale(float(platepar.elev)))

    azim = np.radians(azim)
    alt = np.radians(alt)

    return np.column_stack([np.cos(alt)*np.sin(azim), np.cos(alt)*np.cos(azim), np.sin(alt)])


class TangentFrame(object):
    def __init__(self, platepar, jd, centre, h_px=10.0):
        """ Tangent plane about the optical axis: (tx, ty) = (v.e1/v.c, v.e2/v.c), with e1 along image +x
            and e2 along image +y.

        Arguments:
            platepar: [Platepar object]
            jd: [float] Julian date.
            centre: [tuple] (x, y) pixel of the optical axis.

        Keyword arguments:
            h_px: [float] Pixel offset used to find the image axes on the sky.
        """

        cx, cy = centre

        v = pixelsToDirections(platepar, [cx, cx + h_px, cx, cx - h_px, cx],
            [cy, cy, cy + h_px, cy, cy - h_px], jd)
        c, vx, vy, vmx, vmy = v

        self.c = c/np.linalg.norm(c)

        # Image +x defines +tx
        dx = vx - vmx
        e1 = dx - np.dot(dx, self.c)*self.c
        self.e1 = e1/np.linalg.norm(e1)

        # Image +y defines the sign of +ty
        e2 = np.cross(self.c, self.e1)
        if np.dot(vy - vmy, e2) < 0:
            e2 = -e2

        self.e2 = e2


    def project(self, v):
        """ Project unit vectors (N, 3) to tangent coordinates (N, 2). """

        z = v.dot(self.c)

        return np.column_stack([v.dot(self.e1)/z, v.dot(self.e2)/z])


def frameBorderPixels(x_res, y_res, step_px=2.0):
    """ Sample the frame border [0, x_res] x [0, y_res] every step_px as one loop. """

    nx = max(2, int(round(x_res/step_px)))
    ny = max(2, int(round(y_res/step_px)))

    xs = np.linspace(0, x_res, nx + 1)
    ys = np.linspace(0, y_res, ny + 1)

    x = np.concatenate([xs[:-1], np.full(ny, x_res), xs[::-1][:-1], np.full(ny, 0.0)])
    y = np.concatenate([np.full(nx, 0.0), ys[:-1], np.full(nx, y_res), ys[::-1][:-1]])

    return x, y


def maskRegionsPixels(mask_img, min_area_px):
    """ Masked region (mask == 0) as a shapely geometry in pixel coordinates, with holes. Pixel i covers
        [i, i + 1), so the frame spans [0, X_res] x [0, Y_res].

    Arguments:
        mask_img: [ndarray] Mask image, 0 = masked.
        min_area_px: [int] Masked blobs smaller than this are dropped.

    Return:
        region: [shapely geometry]
    """

    shapely, _ = _importGeometryPackages()

    masked = (mask_img == 0).astype(np.uint8)

    # Drop small blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(masked, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            masked[labels == i] = 0

    contours, hierarchy = cv2.findContours(masked, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if hierarchy is None:
        return shapely.geometry.MultiPolygon()

    hierarchy = hierarchy[0]

    polygons = []
    for i, contour in enumerate(contours):

        # Holes are handled with their parent
        if hierarchy[i][3] != -1:
            continue

        outer = contour[:, 0, :].astype(float) + 0.5

        if len(outer) < 3:
            continue

        holes = []
        j = hierarchy[i][2]
        while j != -1:
            hole = contours[j][:, 0, :].astype(float) + 0.5
            if len(hole) >= 3:
                holes.append(hole)
            j = hierarchy[j][0]

        polygon = shapely.geometry.Polygon(outer, holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        polygons.append(polygon)

    region = shapely.ops.unary_union(polygons)

    # Contours run through pixel centres, the pixels extend half a pixel beyond them
    region = region.buffer(0.5, join_style=2)

    # Straight pixel edges are curves in the tangent plane
    region = shapely.segmentize(region, 2.0)

    return region


def _asPolygons(geom):
    """ List of the polygons in a shapely geometry. """

    if geom.is_empty:
        return []

    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if (g.geom_type == "Polygon") and (g.area > 0)]

    return [geom] if geom.geom_type == "Polygon" else []


def occluderFromMask(mask_img, platepar, distance=150.0, pupil=4.2, margin=0.5, axis="distortion",
    min_area_px=200, simplify=0.2, min_piece=4.0, jd=None):
    """ Compute the occluder outlines at a plane in front of the lens.

    Arguments:
        mask_img: [ndarray] Mask image (Y_res, X_res), 0 = direction to block.
        platepar: [Platepar object]

    Keyword arguments:
        distance: [float] Distance of the plane from the entrance pupil (mm). At 150 mm the few mm between
            the lens front and the pupil do not matter.
        pupil: [float] Entrance pupil diameter (mm), focal length divided by the f-number.
        margin: [float] Extra radius on the pupil disk (mm).
        axis: [str] "distortion" - optical axis at the distortion centre of the platepar (the lens
            barrel), "image" - at the image centre.
        min_area_px: [int] Masked blobs smaller than this are dropped.
        simplify: [float] Polyline tolerance at the plane (mm).
        min_piece: [float] Pieces smaller than this area are dropped (mm^2).
        jd: [float] Julian date at which the platepar is evaluated. None for the platepar reference time.

    Return:
        layers: [dict] Shapely geometry per DXF layer name (see DXF_LAYERS).
    """

    shapely, _ = _importGeometryPackages()

    if mask_img.ndim > 2:
        mask_img = mask_img[:, :, 0]

    if mask_img.shape != (platepar.Y_res, platepar.X_res):
        raise ValueError("The mask ({:d}x{:d}) does not match the platepar ({:d}x{:d})".format(
            mask_img.shape[1], mask_img.shape[0], platepar.X_res, platepar.Y_res))

    if jd is None:
        jd = float(platepar.JD)

    if axis == "distortion":
        centre = [float(c) for c in platepar.getDistortionCentre()]
    else:
        centre = [platepar.X_res/2.0, platepar.Y_res/2.0]

    frame = TangentFrame(platepar, jd, centre)
    radius = pupil/2.0 + margin


    def projectRing(coords):
        xy = np.array(coords)[:-1]
        return frame.project(pixelsToDirections(platepar, xy[:, 0], xy[:, 1], jd))

    def projectPolygon(polygon):
        projected = shapely.geometry.Polygon(projectRing(polygon.exterior.coords),
            [projectRing(hole.coords) for hole in polygon.interiors])
        return projected if projected.is_valid else projected.buffer(0)

    def dropSmall(geom):
        return shapely.ops.unary_union([p for p in _asPolygons(geom) if p.area >= min_piece])


    # Frame border and masked region in tangent coordinates
    border_x, border_y = frameBorderPixels(platepar.X_res, platepar.Y_res)
    frame_t = shapely.geometry.Polygon(frame.project(pixelsToDirections(platepar, border_x, border_y, jd)))
    if not frame_t.is_valid:
        frame_t = frame_t.buffer(0)

    region_px = maskRegionsPixels(mask_img, min_area_px)
    region_t = shapely.ops.unary_union([projectPolygon(p) for p in _asPolygons(region_px)])

    # Scale to the plane and apply the pupil disk
    frame_mm = shapely.affinity.scale(frame_t, xfact=distance, yfact=distance, origin=(0, 0))
    region_mm = shapely.affinity.scale(region_t, xfact=distance, yfact=distance, origin=(0, 0))

    clear = shapely.geometry.polygon.orient(frame_mm.buffer(radius, quad_segs=64), sign=1.0)
    occluder_full = region_mm.buffer(radius, quad_segs=32)
    occluder_safe = dropSmall(region_mm.buffer(-radius, quad_segs=32))

    # Along the frame edge the two buffered boundaries coincide, a small extra buffer on the blocking side
    #   avoids slivers in the plate cutout
    opening_full = dropSmall(clear.difference(occluder_full.buffer(0.2)))
    opening_safe = dropSmall(clear.difference(occluder_safe))

    layers = {"CLEAR": clear, "OCCLUDER_FULL": occluder_full, "OCCLUDER_SAFE": occluder_safe,
        "OPENING_FULL": opening_full, "OPENING_SAFE": opening_safe}

    for name in layers:
        layers[name] = layers[name].simplify(simplify, preserve_topology=True)

    return layers


def writeOccluderDXF(file_path, layers):
    """ Write the occluder outlines as closed polylines, one DXF layer per outline, units mm.

    Arguments:
        file_path: [str] Path of the DXF file.
        layers: [dict] Output of occluderFromMask.
    """

    _, ezdxf = _importGeometryPackages()
    from ezdxf import units

    doc = ezdxf.new("R2010")
    doc.units = units.MM
    doc.header["$INSUNITS"] = units.MM
    msp = doc.modelspace()

    for i, name in enumerate(DXF_LAYERS):
        doc.layers.add(name=name, color=(i%6) + 1)
        for polygon in _asPolygons(layers[name]):
            for ring in [polygon.exterior] + list(polygon.interiors):
                msp.add_lwpolyline(list(ring.coords)[:-1], close=True, dxfattribs={"layer": name})

    doc.layers.add(name="AXIS", color=8)
    msp.add_line((-5, 0), (5, 0), dxfattribs={"layer": "AXIS"})
    msp.add_line((0, -5), (0, 5), dxfattribs={"layer": "AXIS"})

    doc.saveas(file_path)


def occluderSummary(layers, distance):
    """ One-paragraph description of the outlines. """

    def size(geom):
        if geom.is_empty:
            return "empty"
        x_min, y_min, x_max, y_max = geom.bounds
        return "{:.0f} x {:.0f} mm".format(x_max - x_min, y_max - y_min)

    clear = layers["CLEAR"]
    full = layers["OCCLUDER_FULL"].intersection(clear)
    safe = layers["OCCLUDER_SAFE"].intersection(clear)

    return ("Plane at {:.0f} mm: clear opening {:s}.\n"
        "OCCLUDER_FULL: {:s} in {:d} piece(s), {:.0f}% of the clear area.\n"
        "OCCLUDER_SAFE: {:s} in {:d} piece(s), {:.0f}% of the clear area.").format(distance, size(clear),
        size(full), len(_asPolygons(layers["OCCLUDER_FULL"])), 100.0*full.area/clear.area,
        size(safe), len(_asPolygons(layers["OCCLUDER_SAFE"])), 100.0*safe.area/clear.area)


def exportOccluderDXF(file_path, mask_img, platepar, **kwargs):
    """ Compute the occluder outlines and write them to a DXF file.

    Arguments:
        file_path: [str] Path of the DXF file.
        mask_img: [ndarray] Mask image, 0 = direction to block.
        platepar: [Platepar object]

    Keyword arguments:
        See occluderFromMask.

    Return:
        summary: [str] Description of the outlines.
    """

    layers = occluderFromMask(mask_img, platepar, **kwargs)

    if layers["OCCLUDER_FULL"].is_empty:
        raise ValueError("The mask has no masked region to block.")

    writeOccluderDXF(file_path, layers)

    return occluderSummary(layers, kwargs.get("distance", 150.0))



if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="""Make a DXF outline of a physical light occluder
        from a mask image and a platepar.""")

    arg_parser.add_argument('platepar', metavar='PLATEPAR', type=str,
        help="Path to the platepar file.")

    arg_parser.add_argument('mask', metavar='MASK', type=str,
        help="Path to the mask image, 0 = direction to block.")

    arg_parser.add_argument('-d', '--distance', metavar='MM', type=float, default=150.0,
        help="Distance of the occluder plane from the lens (mm). 150 by default.")

    arg_parser.add_argument('-p', '--pupil', metavar='MM', type=float, default=4.2,
        help="Entrance pupil diameter (mm), focal length divided by the f-number. 4.2 by default.")

    arg_parser.add_argument('-m', '--margin', metavar='MM', type=float, default=0.5,
        help="Extra radius on the pupil disk (mm). 0.5 by default.")

    arg_parser.add_argument('-a', '--axis', choices=["distortion", "image"], default="distortion",
        help="Optical axis at the distortion centre of the platepar (default) or at the image centre.")

    arg_parser.add_argument('-o', '--output', metavar='DXF', type=str, default=None,
        help="Path of the DXF file. By default occluder_<station>_<distance>mm.dxf next to the mask.")

    cml_args = arg_parser.parse_args()

    #########################

    platepar = Platepar()
    if not platepar.read(cml_args.platepar):
        sys.exit("Cannot read the platepar: " + cml_args.platepar)

    mask_img = cv2.imread(cml_args.mask, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        sys.exit("Cannot read the mask: " + cml_args.mask)

    dxf_path = cml_args.output
    if dxf_path is None:
        dxf_path = os.path.join(os.path.dirname(os.path.abspath(cml_args.mask)),
            "occluder_{:s}_{:.0f}mm.dxf".format(str(platepar.station_code), cml_args.distance))

    print(exportOccluderDXF(dxf_path, mask_img, platepar, distance=cml_args.distance, pupil=cml_args.pupil,
        margin=cml_args.margin, axis=cml_args.axis))

    print("DXF saved to:", dxf_path)
