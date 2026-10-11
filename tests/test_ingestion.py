"""Tests for validation, reprojection and road snapping (offline, synthetic graph)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import shapely

from src.ingestion import (CollisionLoader, IngestionPipeline, RoadNetworkLoader, RoadSnapper,
                           SchemaError, SpatialProjector, parse_maxspeed_mph)


# --------------------------- validation ----------------------------------- #
def test_validator_rejects_defects(sample_df: pd.DataFrame) -> None:
    valid, rejected = CollisionLoader().validate(sample_df)
    assert len(valid) + len(rejected) == len(sample_df)
    assert {"null_coordinates", "zero_coordinates", "duplicate_id"} <= set(rejected["reject_reason"])
    assert valid["collision_id"].is_unique
    assert valid[["Latitude", "Longitude"]].notna().all().all()


def test_validator_missing_column() -> None:
    with pytest.raises(SchemaError):
        CollisionLoader().validate(pd.DataFrame({"Latitude": [34.0]}))


# --------------------------- projection ----------------------------------- #
def test_projection_to_utm_metres() -> None:
    df = pd.DataFrame({"Latitude": [34.0522], "Longitude": [-118.2437]})  # LA City Hall
    gdf = SpatialProjector().to_projected(df)
    assert gdf.crs.to_epsg() == 32611
    # Known UTM 11N coordinates for downtown LA (≈385 km E, ≈3,768 km N)
    assert abs(gdf["x_utm"].iloc[0] - 385_000) < 3_000
    assert abs(gdf["y_utm"].iloc[0] - 3_768_500) < 3_000


def test_projection_catches_swapped_coordinates() -> None:
    df = pd.DataFrame({"Latitude": [-118.24], "Longitude": [34.05]})
    with pytest.raises(Exception):
        SpatialProjector().to_projected(df)


# --------------------------- snapping ------------------------------------- #
@pytest.fixture(scope="module")
def projected_graph(grid_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    return RoadNetworkLoader().project(grid_graph)


@pytest.fixture(scope="module")
def snapped(projected_graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"collision_id": [f"T{i}" for i in range(400)],
                       "Latitude": rng.uniform(34.005, 34.075, 400),
                       "Longitude": rng.uniform(-118.295, -118.205, 400)})
    pts = SpatialProjector().to_projected(df)
    return RoadSnapper(projected_graph).snap(pts)


def test_snapper_requires_projected_graph(grid_graph: nx.MultiDiGraph) -> None:
    with pytest.raises(ValueError):
        RoadSnapper(grid_graph)


def test_snapped_points_lie_on_their_edge(snapped: gpd.GeoDataFrame,
                                          projected_graph: nx.MultiDiGraph) -> None:
    edges = RoadSnapper(projected_graph).edges
    keys = list(zip(snapped["edge_u"], snapped["edge_v"], snapped["edge_key"]))
    lines = edges.loc[keys, "geometry"].to_numpy()
    assert np.allclose(shapely.distance(snapped.geometry.to_numpy(), lines), 0, atol=1e-6)


def test_snap_distance_matches_displacement(snapped: gpd.GeoDataFrame) -> None:
    moved = snapped.geometry.distance(snapped["raw_geometry"])
    assert np.allclose(moved, snapped["snap_distance_m"], atol=1e-2)
    # Grid spacing ≈ 400 m → nobody is further than half a block from a road
    assert snapped["snap_distance_m"].max() < 250


def test_snap_is_nearest_edge(snapped: gpd.GeoDataFrame, projected_graph: nx.MultiDiGraph) -> None:
    edges = RoadSnapper(projected_graph).edges
    sample = snapped.iloc[:25]
    brute = [edges.geometry.distance(p).min() for p in sample["raw_geometry"]]
    assert np.allclose(brute, sample["snap_distance_m"], atol=1e-2)


def test_edge_attributes_normalised(snapped: gpd.GeoDataFrame) -> None:
    assert set(snapped["speed_limit_mph"].dropna().unique()) <= {30.0, 35.0}
    assert snapped["road_class"].isin(["primary", "residential"]).all()
    assert snapped["lanes"].eq(2).all()
    assert snapped["edge_position"].between(0, 1).all()


@pytest.mark.parametrize("raw, expected", [
    ("35 mph", 35.0), (["25 mph", "35 mph"], 35.0), ("50", 31.1), (None, None), ("signals", None),
])
def test_parse_maxspeed(raw, expected) -> None:
    assert parse_maxspeed_mph(raw) == expected


# --------------------------- end to end ----------------------------------- #
def test_pipeline_end_to_end(tmp_path: Path, grid_graph: nx.MultiDiGraph) -> None:
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "collision_id": [f"E{i}" for i in range(n)],
        "Latitude": rng.uniform(34.005, 34.075, n),
        "Longitude": rng.uniform(-118.295, -118.205, n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
    })
    df.loc[0, ["Latitude", "Longitude"]] = 0.0
    df.loc[1, "Longitude"] = -118.60                 # inside bbox, ~30 km from grid
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)

    res = IngestionPipeline().run(path, graph=grid_graph, write=False)
    reasons = res.rejected.set_index("collision_id")["reject_reason"]
    assert reasons["E0"] == "zero_coordinates"
    assert reasons["E1"] == "no_road_within_tolerance"
    assert res.stats["output_rows"] == n - 2
    assert res.data.crs.to_epsg() == 32611
