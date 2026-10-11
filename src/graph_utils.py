"""Helpers shared by the generator, feature and clustering modules for OSMnx graphs."""

from __future__ import annotations

from typing import Dict, List

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# Relative traffic load by OSM road class (residential = 1).
ROAD_CLASS_LOAD: Dict[str, float] = {
    "motorway": 6.0, "motorway_link": 3.0, "trunk": 4.5, "trunk_link": 2.5,
    "primary": 3.5, "primary_link": 2.0, "secondary": 2.6, "secondary_link": 1.6,
    "tertiary": 1.8, "tertiary_link": 1.3, "unclassified": 1.1, "residential": 1.0,
    "living_street": 0.5, "service": 0.6,
}
# Free-flow fallback when OSM has no maxspeed tag (mph, US urban defaults).
ROAD_CLASS_SPEED_MPH: Dict[str, float] = {
    "motorway": 65, "motorway_link": 45, "trunk": 55, "trunk_link": 40, "primary": 40,
    "primary_link": 35, "secondary": 35, "secondary_link": 30, "tertiary": 30,
    "tertiary_link": 25, "unclassified": 25, "residential": 25, "living_street": 15,
    "service": 15,
}
ARTERIAL_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary")


def first_tag(v: object) -> object:
    """OSM tags may be scalars or lists after simplification; return the first value."""
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def ensure_street_count(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add the OSMnx ``street_count`` node attribute if the graph lacks it."""
    if not all("street_count" in d for _, d in G.nodes(data=True)):
        nx.set_node_attributes(G, ox.stats.count_streets_per_node(G), "street_count")
    return G


def edges_gdf(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Edges with geometry filled and a normalised ``road_class`` / ``road_name``."""
    e = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
    e["road_class"] = e["highway"].map(first_tag).astype("string") if "highway" in e else "unknown"
    e["road_name"] = e["name"].map(first_tag).astype("string") if "name" in e else pd.NA
    return e


def intersections_gdf(G: nx.MultiDiGraph, min_legs: int = 3) -> gpd.GeoDataFrame:
    """
    Junction nodes (``street_count >= min_legs``) with ``legs`` and ``class_mix``
    (number of distinct road classes meeting there).
    """
    ensure_street_count(G)
    nodes = ox.graph_to_gdfs(G, edges=False)
    e = edges_gdf(G).reset_index()
    ends = pd.concat([e[["u", "road_class"]].rename(columns={"u": "node"}),
                      e[["v", "road_class"]].rename(columns={"v": "node"})])
    mix = ends.groupby("node")["road_class"].nunique()
    top = ends.assign(load=ends["road_class"].map(ROAD_CLASS_LOAD).fillna(1.0)) \
              .groupby("node")["load"].max()
    nodes = nodes.assign(legs=nodes["street_count"].astype(int),
                         class_mix=mix.reindex(nodes.index).fillna(1).astype(int),
                         max_load=top.reindex(nodes.index).fillna(1.0))
    return nodes[nodes["legs"] >= min_legs][["legs", "class_mix", "max_load", "geometry"]]


def walk_named_road(G: nx.MultiDiGraph, start: tuple, target_m: float,
                    rng: np.random.Generator) -> List[tuple]:
    """
    Follow edges that share the start edge's road name until ``target_m`` metres,
    never reversing. Returns the list of (u, v, key) edges walked.
    """
    u, v, k = start
    name = first_tag(G.edges[u, v, k].get("name"))
    path, length, prev = [start], float(G.edges[u, v, k].get("length", 0.0)), u
    seen = {u, v}
    while length < target_m and name is not None:
        nxt = [(v, w, kk) for _, w, kk, d in G.out_edges(v, keys=True, data=True)
               if w != prev and w not in seen and first_tag(d.get("name")) == name]
        if not nxt:
            break
        step = nxt[rng.integers(len(nxt))]
        path.append(step)
        length += float(G.edges[step].get("length", 0.0))
        prev, v = v, step[1]
        seen.add(v)
    return path
