"""
End-to-end tests: train a small model on the offline test grid, serve it with
FastAPI's TestClient, and check that serving reproduces training features.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app as api_module
from config import settings
from src.data_generator import CollisionDataGenerator
from src.features import FeatureEngineeringPipeline
from src.inference import RiskScorer, assign_tier
from src.ingestion import IngestionPipeline, RoadNetworkLoader
from src.train import TrainingPipeline, make_target, optimise_threshold

api_module.state.autoload = False      # never load the on-disk bundle / download OSM in tests

GRID_BOX = (-118.295, 34.005, -118.205, 34.075)


@pytest.fixture(scope="module")
def grid_features(grid_graph, tmp_path_factory) -> gpd.GeoDataFrame:
    """Generator output planted on the test grid → ingestion → features."""
    df = CollisionDataGenerator(seed=11).generate(8_000, graph=grid_graph)
    w, s, e, n = GRID_BOX
    df = df[df["Longitude"].between(w, e) & df["Latitude"].between(s, n)]
    path = tmp_path_factory.mktemp("api") / "raw.csv"
    df.to_csv(path, index=False)
    snapped = IngestionPipeline().run(path, graph=grid_graph, write=False).data
    G = RoadNetworkLoader().project(grid_graph)
    return FeatureEngineeringPipeline(G).run(snapped, write=False).data


@pytest.fixture(scope="module")
def trained(grid_graph, grid_features):
    return TrainingPipeline(grid_graph, "underwriting",
                            models=("logistic_regression", "xgboost")).run(grid_features,
                                                                           write=False)


@pytest.fixture(scope="module")
def scorer(trained, grid_graph) -> RiskScorer:
    return RiskScorer(trained.bundle, grid_graph)


@pytest.fixture(scope="module")
def client(scorer):
    api_module.state.scorer, api_module.state.error = scorer, None
    with TestClient(api_module.app) as c:
        yield c
    api_module.state.scorer = None


def scenario(**kw) -> dict:
    base = {"latitude": 34.04, "longitude": -118.25, "timestamp": "2025-11-14T22:30:00",
            "driver_age": 22, "vehicle_type": "motorcycle", "vehicle_year": 2016,
            "prior_claims_3yr": 1, "weather": "rain"}
    return {**base, **kw}


# ------------------------------- training --------------------------------- #
def test_target_definition(grid_features) -> None:
    y = make_target(grid_features)
    assert set(y.unique()) <= {0, 1}
    assert y.mean() == pytest.approx((grid_features["severity_code"] >= 2).mean())
    assert 0.05 < y.mean() < 0.13


def test_training_outputs(trained) -> None:
    m = trained.metrics
    assert set(m["model"]) == {"logistic_regression", "xgboost"}
    for col in ("test_pr_auc", "test_roc_auc", "fixed_0.20_recall", "optimised_recall"):
        assert m[col].between(0, 1).all()
    assert trained.chosen_model == m.sort_values("oof_pr_auc").iloc[-1]["model"]
    assert len(trained.tier_cutoffs) == 3 and np.all(np.diff(trained.tier_cutoffs) > 0)
    for d in trained.details.values():
        assert set(d["thresholds"]) == {"default_0.50", "fixed_0.20", "optimised"}


def test_smote_only_touches_training_folds(trained) -> None:
    """The fitted pipeline resamples in fit only; predict sees real rows unchanged."""
    pipe = trained.bundle["model"]
    assert "smote" in pipe.named_steps
    X = pd.DataFrame({c: [0.0] for c in trained.bundle["builder"].numeric_cols})
    for c in trained.bundle["builder"].categorical_cols:
        X[c] = "missing"
    X["cluster_id"] = -1
    assert pipe.predict_proba(X).shape == (1, 2)               # 1 in → 1 out


def test_threshold_respects_flag_cap() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(5_000) < 0.1
    p = rng.random(5_000)                                   # no signal at all
    t, table = optimise_threshold(y.astype(int), p)
    assert (p >= t).mean() <= settings.model.max_flag_share + 0.02


def test_assign_tier_boundaries() -> None:
    """A probability equal to a cut point moves up a tier (same rule as p >= threshold)."""
    tiers = assign_tier(np.array([0.0, 0.0999, 0.1, 0.5, 0.6, 0.99]), [0.1, 0.3, 0.6])
    assert tiers.tolist() == ["Low", "Low", "Moderate", "High", "Very High", "Very High"]


# --------------------------- train/serve parity --------------------------- #
def test_serving_reproduces_training_predictions(trained, scorer, grid_features) -> None:
    """Scoring raw records through the API path must equal the training matrix path."""
    raw = grid_features.sample(200, random_state=0)
    cols = ["Latitude", "Longitude", "timestamp", "driver_age", "vehicle_type", "vehicle_year",
            "prior_claims_3yr", "weather", "road_surface", "lighting"]
    served = scorer.score(pd.DataFrame(raw[cols]))["probability_high_risk"].to_numpy()

    # Training path for the same rows
    from src.inference import add_derived
    m = scorer.matcher
    X = np.column_stack([raw.geometry.x, raw.geometry.y])
    hot = m.describe(m.match(X)).set_index(raw.index)
    df = add_derived(pd.concat([raw.drop(columns=[c for c in hot if c in raw]), hot], axis=1))
    expected = scorer.model.predict_proba(scorer.builder.transform(df))[:, 1]
    # area_name is re-derived from the snapped point; a crash near an LAPD
    # boundary can flip area, so allow a handful of tiny differences.
    assert np.mean(np.isclose(served, expected, atol=1e-9)) >= 0.97


# ---------------------------------- API ----------------------------------- #
def test_health_and_info(client) -> None:
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["feature_set"] == "underwriting" and len(info["tier_cutoffs"]) == 3


def test_predict_happy_path(client) -> None:
    r = client.post("/predict", json=scenario())
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability_high_risk"] <= 1
    assert body["risk_tier"] in settings.model.tier_labels
    assert body["high_risk"] == (body["probability_high_risk"] >= body["decision_threshold"])
    loc = body["location"]
    assert loc["snap_distance_m"] < 250 and loc["road_name"]
    assert loc["hotspot_type"] in {"intersection", "corridor", "area", "none"}


def test_predict_is_deterministic_and_batch_consistent(client) -> None:
    a = client.post("/predict", json=scenario()).json()
    b = client.post("/predict", json=scenario()).json()
    assert a == b
    other = scenario(latitude=34.06, longitude=-118.22, vehicle_type="sedan")
    batch = client.post("/predict/batch", json={"scenarios": [other, scenario()]}).json()
    assert batch["predictions"][1]["prediction"] == a        # position in batch irrelevant


def test_batch_isolates_off_network_rows(client) -> None:
    far = scenario(latitude=34.30, longitude=-118.60)
    r = client.post("/predict/batch", json={"scenarios": [scenario(), far, scenario()]})
    assert r.status_code == 200
    items = r.json()["predictions"]
    assert items[0]["prediction"] and items[2]["prediction"]
    assert items[1]["prediction"] is None and "from any road" in items[1]["error"]
    assert items[0]["prediction"] == items[2]["prediction"]


def test_timezone_handling(client) -> None:
    naive = client.post("/predict", json=scenario(timestamp="2025-07-01T03:00:00")).json()
    utc = client.post("/predict", json=scenario(timestamp="2025-07-01T10:00:00Z")).json()
    assert naive == utc                                      # 10:00Z = 03:00 PDT


def test_rejects_protected_attribute(client) -> None:
    r = client.post("/predict", json=scenario(driver_gender="M"))
    assert r.status_code == 422


def test_rejects_outside_study_area(client) -> None:
    assert client.post("/predict", json=scenario(latitude=40.7, longitude=-74.0)).status_code == 422


def test_rejects_off_network_location(client) -> None:
    r = client.post("/predict", json=scenario(latitude=34.30, longitude=-118.60))
    assert r.status_code == 422 and "from any road" in r.json()["detail"]


def test_claims_triage_requires_post_crash_fields(trained, grid_graph) -> None:
    bundle = dict(trained.bundle, feature_set="claims_triage")
    api_module.state.scorer = RiskScorer(bundle, grid_graph)
    try:
        with TestClient(api_module.app) as c:
            r = c.post("/predict", json=scenario())
        assert r.status_code == 422 and "collision_type" in r.json()["detail"]
    finally:
        api_module.state.scorer = None


def test_unavailable_when_model_missing() -> None:
    api_module.state.scorer, api_module.state.error = None, "FileNotFoundError: x"
    with TestClient(api_module.app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=scenario()).status_code == 503
    api_module.state.error = None
