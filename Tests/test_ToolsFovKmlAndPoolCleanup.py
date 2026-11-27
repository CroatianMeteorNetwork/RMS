""" Tests for fovKML empty mask polygons and QueuedPool Manager cleanup in TrackStack and FluxBatch. """

from __future__ import print_function, division, absolute_import

import ast
import os
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

import Utils.FOVKML as fovkml_mod


@pytest.mark.parametrize("sides", [
    [],
    [[], [], [], []],
    [[], [[43.1, -81.1, 100000.0], [43.2, -81.0, 100000.0]], [], []],
])
@pytest.mark.parametrize("plot_station", [True, False])
def testFovKMLEmptySidesAndPolygons(tmp_path, monkeypatch, sides, plot_station):
    """ fovKML must write a valid KML when mask coverage leaves some or all FOV sides empty. """

    pp = SimpleNamespace(station_code="XX0001", lat=43.0, lon=-81.0, elev=250.0)
    monkeypatch.setattr(fovkml_mod, "fovArea", lambda platepar, mask, area_ht, side_points: sides)

    kml_path = fovkml_mod.fovKML(str(tmp_path), pp, plot_station=plot_station)
    assert os.path.isfile(kml_path)

    with open(kml_path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    # Output must parse as well-formed XML
    root = ET.fromstring(xml_text)
    assert root.tag.endswith("kml")

    # Non-empty sides produce at least one Polygon; empty sides produce none
    has_any_side = any(len(s) > 0 for s in sides)
    assert ("<Polygon>" in xml_text) is has_any_side


@pytest.mark.parametrize("rel_path", [
    "Utils/TrackStack.py",
    "Utils/FluxBatch.py",
])
def testToolScriptsShutDownQueuedPoolManagerInFinally(rel_path):
    """ TrackStack and FluxBatch must call shutdownManager() inside a finally block after QueuedPool use. """

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(repo_root, rel_path)
    with open(src_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=rel_path)

    guarded_pools = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            has_start = any(
                isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "startPool"
                for stmt in node.body for sub in ast.walk(stmt)
            )
            has_shutdown = any(
                isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "shutdownManager"
                for stmt in node.finalbody for sub in ast.walk(stmt)
            )
            if has_start and has_shutdown:
                guarded_pools.append(node.lineno)

    assert len(guarded_pools) >= 1, "{} must guard startPool() with finally shutdownManager()".format(rel_path)
