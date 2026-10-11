"""Tests for two-scale DBSCAN hotspot detection on a planted toy city."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from config import settings
from src.clustering import (NOISE, ClusterProfiler, DBSCANTuner, HotspotClusteringPipeline,
                            HotspotRecoveryEvaluator, KDistanceAnalyzer, cluster_shapes)

X0, Y0 = 370_000.0, 3_760_000.0       # UTM 11N, west LA


@pytest.fixture(scope="module")
def toy_city() -> gpd.GeoDataFrame:
    """12 junction knots, 6 straight corridors, uniform background on a 10×10 km tile."""
    rng = np.random.default_rng(42)
    rows = []
    knots = rng.uniform([X0 + 500, Y0 + 500], [X0 + 9_500, Y0 + 9_500], size=(12, 2))
    for i, (cx, cy) in enumerate(knots):
        for _ in range(40):
            rows.append((cx + rng.normal(0, 8), cy + rng.normal(0, 8), f"INT-{i}",
                         "intersection", f"Road {i}", 5.0))
    for i in range(6):
        sx, sy = rng.uniform(X0 + 1_000, X0 + 7_000), rng.uniform(Y0 + 1_000, Y0 + 9_000)
        horizontal = i % 2 == 0
        t = rng.uniform(0, 1_500, 120)
        for tt in t:
            x = sx + tt if horizontal else sx + rng.normal(0, 3)
            y = sy + rng.normal(0, 3) if horizontal else sy + tt
            rows.append((x, y, f"COR-{i}", "corridor", f"Ave {i}", 200.0))
    for _ in range(3_000):
        rows.append((rng.uniform(X0, X0 + 10_000), rng.uniform(Y0, Y0 + 10_000), "none",
                     "background", None, 200.0))
    df = pd.DataFrame(rows, columns=["x", "y", "gt_hotspot_id", "gt_hotspot_type",
                                     "road_name", "dist_to_intersection_m"])
    df["severity_code"] = rng.integers(0, 4, len(df))
    df["claim_amount_usd"] = rng.lognormal(9, 1, len(df))
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:32611")


@pytest.fixture(scope="module")
def result(toy_city):
    return HotspotClusteringPipeline().run(toy_city, write=False)


def test_cluster_shapes_line_vs_blob() -> None:
    rng = np.random.default_rng(0)
    line = np.c_[np.linspace(0, 1000, 200), rng.normal(0, 2, 200)]
    blob = rng.normal(0, 10, (200, 2))
    X = np.vstack([line, blob])
    labels = np.r_[np.zeros(200, int), np.ones(200, int)]
    s = cluster_shapes(X, labels, 2)
    assert s.loc[0, "linearity"] > 0.99 and s.loc[0, "length_m"] == pytest.approx(1000, rel=0.05)
    assert s.loc[1, "linearity"] < 0.3


def test_pai_score_exact() -> None:
    tuner = DBSCANTuner(study_area_m2=1_000_000.0)                 # 1 km² city
    X = np.array([[0, 0], [1, 1], [2, 2], [500, 500]], float)       # 3 in one 25 m cell
    labels = np.array([0, 0, 0, NOISE])
    sc = tuner.score(labels, tuner._cells(X), n_total=4)
    assert sc["coverage"] == 0.75
    assert sc["pai"] == pytest.approx(0.75 / (625 / 1e6), rel=1e-6)


def test_small_clusters_returned_to_noise() -> None:
    tuner = DBSCANTuner(study_area_m2=1.0)
    lab = np.r_[np.zeros(15, int), np.ones(4, int), [NOISE]]
    out = tuner._drop_small(lab)
    assert (out[:15] == 0).all() and (out[15:] == NOISE).all()


def test_knee_between_hotspot_and_background(toy_city) -> None:
    X = np.c_[toy_city.geometry.x, toy_city.geometry.y]
    knee = KDistanceAnalyzer(k=8).fit(X)
    assert 5 < knee.knee_eps_m < 600
    assert np.all(np.diff(knee.distances) >= 0)


def test_recovers_planted_hotspots(result) -> None:
    ev = result.evaluation
    assert ev["intersection_recall"] == 1.0
    assert ev["corridor_recall"] >= 5 / 6
    assert ev["cluster_precision"] >= 0.9
    assert ev["background_in_hotspots"] < 0.03
    assert ev["type_accuracy"] >= 0.8      # a corridor crossing a knot can merge into "area"


def test_cluster_types_and_summary(result) -> None:
    c = result.clusters
    assert {"intersection", "corridor"} <= set(c["cluster_type"])
    cor = c[c["cluster_type"] == "corridor"]
    assert (cor["length_m"] >= 4 * cor["eps_m"]).all()
    assert (cor["local_lift"] >= settings.clustering.corridor_min_local_lift).all()
    ints = c[c["cluster_type"] == "intersection"]
    assert (ints["length_m"] <= settings.clustering.intersection_max_length_m).all()
    assert c["loss_rank"].tolist() == sorted(c["loss_rank"])          # sorted by loss
    assert c.crs.to_epsg() == 32611 and c.geometry.is_valid.all()


def test_collision_labels_consistent(result) -> None:
    d = result.data
    assert ((d["cluster_id"] == NOISE) == (d["cluster_type"] == "noise")).all()
    assert d.loc[d["in_hotspot"] == 1, "hotspot_loss_rank"].notna().all()
    assert set(d["cluster_stage"]) <= {"intersection", "corridor", "noise"}


def test_rejects_geographic_crs(toy_city) -> None:
    with pytest.raises(ValueError):
        HotspotClusteringPipeline().run(toy_city.to_crs("EPSG:4326"), write=False)


def test_evaluator_counts_fragments() -> None:
    gt = pd.Series(["A"] * 20 + ["none"] * 5)
    kind = pd.Series(["intersection"] * 20 + ["background"] * 5)
    labels = np.r_[np.zeros(10, int), np.ones(10, int), [NOISE] * 5]
    X = np.zeros((25, 2))
    ev = HotspotRecoveryEvaluator().evaluate(gt, kind, labels, {0: "intersection",
                                                                1: "intersection"}, X)
    assert ev["intersection_recall"] == 1.0 and ev["intersection_fragments"] == 2.0
    assert ev["background_in_hotspots"] == 0.0
