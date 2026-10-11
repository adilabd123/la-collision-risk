"""Shared fixtures: a synthetic OSM-like road graph so tests run offline."""

from __future__ import annotations

from typing import Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import pytest
from shapely.geometry import LineString

from src.data_generator import CollisionDataGenerator


def make_grid_graph(
    bbox: Tuple[float, float, float, float] = (-118.30, 34.00, -118.20, 34.08),
    step_deg: float = 0.004,
) -> nx.MultiDiGraph:
    """
    Build an unprojected (EPSG:4326) two-way street grid that looks like an
    OSMnx graph: node x/y, edge length/highway/maxspeed/lanes/name/osmid, and a
    curved geometry on every third edge to exercise true-geometry snapping.
    """
    w, s, e, n = bbox
    xs, ys = np.arange(w, e + 1e-9, step_deg), np.arange(s, n + 1e-9, step_deg)
    G = nx.MultiDiGraph(crs="EPSG:4326")
    nid = lambda i, j: i * len(ys) + j  # noqa: E731
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            G.add_node(nid(i, j), x=float(x), y=float(y))

    way = 0
    for i in range(len(xs)):
        for j in range(len(ys)):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b[0] >= len(xs) or b[1] >= len(ys):
                    continue
                u, v = nid(*a), nid(*b)
                p, q = (xs[a[0]], ys[a[1]]), (xs[b[0]], ys[b[1]])
                # Streets are named per grid line so arterials form continuous
                # named corridors; every 5th line is a primary arterial.
                line_no = j if di == 1 else i
                arterial = line_no % 5 == 0
                attrs = {
                    "osmid": way, "name": f"{'Row' if di == 1 else 'Col'} {line_no}",
                    "highway": ["primary", "secondary"] if arterial else "residential",
                    "maxspeed": "35 mph" if way % 2 else ["25 mph", "30 mph"],
                    "lanes": "2", "oneway": False, "reversed": False,
                    "length": float(ox.distance.great_circle(p[1], p[0], q[1], q[0])),
                }
                if way % 3 == 0:  # curved road: bow the midpoint sideways
                    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
                    off = step_deg * 0.15
                    mid = (mx + off * dj, my + off * di)
                    attrs["geometry"] = LineString([p, mid, q])
                for uu, vv in ((u, v), (v, u)):
                    a2 = dict(attrs)
                    if "geometry" in attrs and uu == v:
                        a2["geometry"] = LineString(list(attrs["geometry"].coords)[::-1])
                    G.add_edge(uu, vv, **a2)
                way += 1
    return G


@pytest.fixture(scope="session")
def grid_graph() -> nx.MultiDiGraph:
    return make_grid_graph()


@pytest.fixture(scope="session")
def sample_df():
    """5k synthetic records (generation is deterministic for a given seed)."""
    return CollisionDataGenerator(seed=123).generate(5_000)
