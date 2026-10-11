"""Tests for temporal features, enrichment providers and collinearity screening."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.features import (CollinearityScreener, FeatureEngineeringPipeline,
                          IntersectionComplexityProvider, POIDensityProvider,
                          SimulatedTrafficProvider, SimulatedWeatherProvider,
                          TemporalFeatureExtractor, VenueSource, hash_uniform)
from src.ingestion import IngestionPipeline, RoadNetworkLoader


@pytest.fixture(scope="module")
def projected_graph(grid_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    return RoadNetworkLoader().project(grid_graph)


@pytest.fixture(scope="module")
def snapped(tmp_path_factory, grid_graph, sample_df) -> gpd.GeoDataFrame:
    """Generator output restricted to the test grid, then ingested."""
    w, s, e, n = -118.295, 34.005, -118.205, 34.075
    df = sample_df.dropna(subset=["Latitude"])
    df = df[df["Longitude"].between(w, e) & df["Latitude"].between(s, n)]
    path = tmp_path_factory.mktemp("feat") / "raw.csv"
    df.to_csv(path, index=False)
    return IngestionPipeline().run(path, graph=grid_graph, write=False).data


# ------------------------------ temporal ---------------------------------- #
def test_temporal_columns_and_ranges() -> None:
    ts = pd.Series(pd.to_datetime(["2024-03-04 00:15", "2024-03-09 23:45", "2024-07-10 08:00"]))
    out = TemporalFeatureExtractor().transform(pd.DataFrame({"timestamp": ts}))
    assert out["Collision_Hour"].tolist() == [0, 23, 8]
    assert out["DayOfWeek"].tolist() == [0, 5, 2]
    assert out["Month"].tolist() == [3, 3, 7]
    assert out["IsRushHour"].tolist() == [0, 0, 1]
    assert out["WeekendNight"].tolist() == [0, 1, 0]


def test_cyclical_encoding_wraps_midnight() -> None:
    ts = pd.Series(pd.to_datetime(["2024-01-01 23:59", "2024-01-02 00:01", "2024-01-01 12:00"]))
    out = TemporalFeatureExtractor().transform(pd.DataFrame({"timestamp": ts}))
    v = out[["Hour_Sin_24h", "Hour_Cos_24h"]].to_numpy()
    assert np.linalg.norm(v[0] - v[1]) < 0.02          # 23:59 ≈ 00:01
    assert np.linalg.norm(v[0] - v[2]) > 1.9           # opposite side of the clock
    assert np.allclose(np.hypot(*v.T), 1.0)


# ------------------------------ enrichment -------------------------------- #
def test_hash_is_deterministic_and_uniform() -> None:
    k = np.arange(100_000)
    a, b = hash_uniform(1, k), hash_uniform(1, k)
    assert np.array_equal(a, b)
    assert 0.49 < a.mean() < 0.51 and a.min() >= 0 and a.max() < 1
    assert not np.array_equal(a, hash_uniform(2, k))


def test_weather_physically_consistent(snapped: gpd.GeoDataFrame) -> None:
    w = SimulatedWeatherProvider().enrich(snapped)
    rain = snapped["weather"] == "rain"
    assert (w.loc[rain, "precip_mm_hr"] > 0).all()
    assert (w.loc[snapped["weather"] == "clear", "precip_mm_hr"] == 0).all()
    wet = snapped["road_surface"] == "wet"
    assert w.loc[wet, "surface_friction_index"].mean() < w.loc[~wet, "surface_friction_index"].mean() - 0.15
    if (snapped["weather"] == "fog").any():
        assert w.loc[snapped["weather"] == "fog", "visibility_km"].max() < 3


def test_traffic_rush_hour_and_repeatability(snapped: gpd.GeoDataFrame) -> None:
    t = SimulatedTrafficProvider().enrich(snapped)
    assert t["congestion_index"].between(0, 1).all()
    assert (t["observed_speed_mph"] <= t["free_flow_speed_mph"] + 1e-9).all()
    hour = pd.to_datetime(snapped["timestamp"]).dt.hour
    weekday = pd.to_datetime(snapped["timestamp"]).dt.dayofweek < 5
    assert t.loc[weekday & hour.isin([8, 17]), "congestion_index"].mean() > \
        t.loc[hour.isin([2, 3]), "congestion_index"].mean() + 0.1
    # Same segment + same hour → same answer (API-like)
    dup = pd.concat([snapped.iloc[:5], snapped.iloc[:5]])
    t2 = SimulatedTrafficProvider().enrich(dup)
    assert np.array_equal(t2.iloc[:5].to_numpy(), t2.iloc[5:].to_numpy())


class _FixedVenues(VenueSource):
    def __init__(self, pts: gpd.GeoSeries) -> None:
        self.pts = pts

    def load(self) -> gpd.GeoSeries:
        return self.pts


def test_poi_counts_exact(snapped: gpd.GeoDataFrame) -> None:
    p0 = snapped.geometry.iloc[0]
    venues = gpd.GeoSeries(gpd.points_from_xy([p0.x + 50, p0.x + 150, p0.x + 199, p0.x + 260],
                                              [p0.y] * 4), crs=snapped.crs)
    out = POIDensityProvider(_FixedVenues(venues)).enrich(snapped.iloc[[0]])
    assert out["alcohol_venues_200m"].iloc[0] == 3
    assert out["dist_nearest_alcohol_venue_m"].iloc[0] == pytest.approx(50, abs=0.1)


def test_intersection_features(snapped: gpd.GeoDataFrame, projected_graph) -> None:
    out = IntersectionComplexityProvider(projected_graph).enrich(snapped)
    assert (out["intersection_legs"] == 4).all()                  # every grid node is a 4-way
    assert out["dist_to_intersection_m"].max() < 260              # ≤ half a block (+ curve)
    assert (out.loc[out["at_intersection"] == 1, "dist_to_intersection_m"] <= 25).all()
    near, far = out["dist_to_intersection_m"] < 25, out["dist_to_intersection_m"] > 120
    assert out.loc[near, "intersection_complexity"].mean() > out.loc[far, "intersection_complexity"].mean()


# ------------------------------ screening --------------------------------- #
def test_collinearity_drops_redundant_keeps_protected() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    df = pd.DataFrame({"a": a, "a_copy": a * 2 + rng.normal(0, 0.01, 2000),
                       "b": rng.normal(size=2000), "const": 1.0,
                       "a_sin": a + rng.normal(0, 0.05, 2000)})
    rep = CollinearityScreener(protected=["a_sin"]).fit(df, list(df.columns))
    assert "const" not in rep.corr                           # constant removed
    assert "a_sin" in rep.kept and "b" in rep.kept
    assert sum(c in rep.kept for c in ("a", "a_copy")) == 0  # both collide with protected a_sin
    assert (rep.high_pairs["r"].abs() > 0.85).all()
    assert rep.vif.max() < 10


def test_collinearity_uses_target_to_choose() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=3000)
    y = x + rng.normal(0, 0.5, 3000)
    df = pd.DataFrame({"good": x, "noisy": x + rng.normal(0, 0.4, 3000)})
    rep = CollinearityScreener().fit(df, ["good", "noisy"], target=pd.Series(y))
    assert rep.dropped == ["noisy"]


def test_pipeline_end_to_end(snapped: gpd.GeoDataFrame, projected_graph, tmp_path) -> None:
    fs = FeatureEngineeringPipeline(projected_graph, providers=None).run(snapped, write=False)
    assert len(fs.data) == len(snapped) and fs.data.crs == snapped.crs
    for col in ("Collision_Hour", "DayOfWeek", "Month"):            # protected by spec
        assert col in fs.feature_columns
    for leak in ("severity_code", "claim_amount_usd", "driver_gender", "gt_hotspot_id"):
        assert leak not in fs.feature_columns
    sel = fs.data[fs.feature_columns].astype(float).corr().abs().to_numpy().copy()
    np.fill_diagonal(sel, 0)
    protected = {"Collision_Hour", "DayOfWeek", "Month", "Hour_Sin_24h", "Hour_Cos_24h",
                 "Hour_Sin_12h", "Hour_Cos_12h", "DoW_Sin", "DoW_Cos", "Month_Sin", "Month_Cos"}
    unprot = [i for i, c in enumerate(fs.feature_columns) if c not in protected]
    assert sel[np.ix_(unprot, unprot)].max() <= 0.85 + 1e-9
