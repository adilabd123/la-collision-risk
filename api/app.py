"""
FastAPI serving layer for the LA Collision Insurance Risk model.

One request = one collision scenario (place, time, conditions, driver, vehicle).
The API snaps the location to the road network, computes the same features as
training, matches the location to a DBSCAN hotspot, runs the model and returns:

* ``probability_high_risk`` – model score that a crash here would be severe/fatal
* ``high_risk`` – that score against the tuned decision threshold
* ``risk_tier`` – 4-tier crash-level risk (Low / Moderate / High / Very High)
* ``location`` – snapped road, hotspot membership and the road segment's own
  spatial risk tier (expected severe crashes per year)

Run locally::

    uvicorn api.app:app --reload

Environment (``CRS_RISK_`` prefix): ``MODEL_PATH``, ``GRAPHML``, ``MAX_BATCH``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Literal, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from config import settings
from src.inference import OffNetworkError, RiskScorer
from src.utils import get_logger

logger = get_logger("api")
_W, _S, _E, _N = settings.study_area.bbox
_TZ = ZoneInfo(settings.api.timezone)
_POST_CRASH = ("collision_type", "num_vehicles", "speeding", "alcohol_involved",
               "distracted", "hit_and_run")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class CollisionScenario(BaseModel):
    """A collision scenario to score. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "latitude": 34.0522, "longitude": -118.2437, "timestamp": "2025-11-14T22:30:00-08:00",
        "driver_age": 22, "vehicle_type": "motorcycle", "vehicle_year": 2016,
        "prior_claims_3yr": 1, "weather": "rain"}})

    latitude: float = Field(..., ge=_S, le=_N, description="WGS84 latitude (City of LA)")
    longitude: float = Field(..., ge=_W, le=_E, description="WGS84 longitude (City of LA)")
    timestamp: Optional[datetime] = Field(
        None, description="ISO-8601; naive times are Los Angeles local. Default: now.")
    driver_age: int = Field(..., ge=16, le=100)
    vehicle_type: Literal["sedan", "suv", "pickup", "van", "commercial_truck", "motorcycle",
                          "bus", "bicycle"]
    vehicle_year: int = Field(..., ge=1980, le=2030)
    prior_claims_3yr: int = Field(0, ge=0, le=20)
    weather: Literal["clear", "cloudy", "rain", "fog"] = "clear"
    road_surface: Optional[Literal["dry", "wet"]] = Field(
        None, description="Default: wet if raining, else dry")
    lighting: Optional[Literal["daylight", "dusk_dawn", "dark_street_lights",
                               "dark_no_street_lights"]] = Field(
        None, description="Default: from sun position at the timestamp, street lights assumed")

    # Post-crash facts — required only by a claims_triage model
    collision_type: Optional[Literal["rear_end", "broadside", "sideswipe", "hit_object",
                                     "pedestrian", "head_on", "other"]] = None
    num_vehicles: Optional[int] = Field(None, ge=1, le=20)
    speeding: Optional[bool] = None
    alcohol_involved: Optional[bool] = None
    distracted: Optional[bool] = None
    hit_and_run: Optional[bool] = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenarios: List[CollisionScenario] = Field(..., min_length=1)


class Location(BaseModel):
    snapped_latitude: float
    snapped_longitude: float
    snap_distance_m: float
    snap_quality: Literal["good", "low_confidence"]
    road_name: Optional[str]
    road_class: Optional[str]
    hotspot_id: Optional[int] = Field(None, description="DBSCAN hotspot id; null if none")
    hotspot_type: Literal["intersection", "corridor", "area", "none"]
    hotspot_loss_rank: Optional[int] = Field(None, description="1 = highest insured loss")
    segment_risk_tier: Optional[str] = Field(
        None, description="Road segment tier; null if the segment had no training crashes")
    segment_expected_severe_per_year: Optional[float] = None


class Prediction(BaseModel):
    probability_high_risk: float
    high_risk: bool
    decision_threshold: float
    risk_tier: Literal["Low", "Moderate", "High", "Very High"]
    location: Location
    model: str
    feature_set: str


class BatchItem(BaseModel):
    index: int
    prediction: Optional[Prediction] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    predictions: List[BatchItem]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
class _State:
    scorer: Optional[RiskScorer] = None
    error: Optional[str] = None
    autoload: bool = True          # tests inject a scorer and switch this off


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model bundle and road graph once at startup."""
    if state.scorer is None and state.autoload:
        try:
            state.scorer = RiskScorer.load()
            logger.info("Model '%s' (%s) loaded", state.scorer.bundle["model_name"],
                        state.scorer.bundle["feature_set"])
        except Exception as exc:                     # keep serving /health with the reason
            state.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Model failed to load")
    yield


app = FastAPI(title="LA Collision Insurance Risk API", version="1.0.0", lifespan=lifespan)


def _scorer() -> RiskScorer:
    if state.scorer is None:
        raise HTTPException(503, detail=f"Model not loaded: {state.error or 'starting up'}")
    return state.scorer


@app.exception_handler(OffNetworkError)
async def _off_network(_: Request, exc: OffNetworkError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict:
    ok = state.scorer is not None
    return JSONResponse(status_code=200 if ok else 503,
                        content={"status": "ok" if ok else "unavailable",
                                 **({} if ok else {"reason": state.error})})


@app.get("/model-info")
def model_info() -> dict:
    s = _scorer()
    b = s.bundle
    return {"model": b["model_name"], "feature_set": b["feature_set"],
            "trained_at": b["trained_at"], "train_rows": b["train_rows"],
            "base_rate": round(b["base_rate"], 4), "decision_threshold": b["threshold"],
            "tier_labels": b["tier_labels"], "tier_cutoffs": [round(c, 4) for c in b["tier_cutoffs"]],
            "hotspot_params": b["hotspot_params"], "metrics": b["metrics"],
            "required_post_crash_fields": list(_POST_CRASH)
            if b["feature_set"] == "claims_triage" else []}


@app.post("/predict", response_model=Prediction)
def predict(scenario: CollisionScenario) -> Prediction:
    return _predict([scenario], off_network="raise")[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    """Score many scenarios; an off-network location fails only its own item."""
    if len(req.scenarios) > settings.api.max_batch:
        raise HTTPException(413, detail=f"At most {settings.api.max_batch} scenarios per batch")
    preds = _predict(req.scenarios, off_network="flag")
    return BatchResponse(predictions=[
        BatchItem(index=i, prediction=p) if isinstance(p, Prediction)
        else BatchItem(index=i, error=p) for i, p in enumerate(preds)])


def _predict(scenarios: List[CollisionScenario], off_network: str):
    """Predictions in request order; with ``off_network='flag'`` failed rows are strings."""
    s = _scorer()
    df = _to_frame(scenarios, s.bundle["feature_set"])
    out = s.score(df, off_network=off_network)
    meta = {"model": s.bundle["model_name"], "feature_set": s.bundle["feature_set"],
            "decision_threshold": s.threshold}
    return [r.error if isinstance(getattr(r, "error", None), str) else Prediction(
        probability_high_risk=round(float(r.probability_high_risk), 4),
        high_risk=bool(r.high_risk), risk_tier=r.risk_tier,
        location=Location(
            snapped_latitude=round(float(r.snapped_latitude), 6),
            snapped_longitude=round(float(r.snapped_longitude), 6),
            snap_distance_m=round(float(r.snap_distance_m), 1), snap_quality=r.snap_quality,
            road_name=_none(r.road_name), road_class=_none(r.road_class),
            hotspot_id=None if r.hotspot_id < 0 else int(r.hotspot_id),
            hotspot_type=r.hotspot_type,
            hotspot_loss_rank=None if pd.isna(r.hotspot_loss_rank) else int(r.hotspot_loss_rank),
            segment_risk_tier=_none(r.segment_risk_tier),
            segment_expected_severe_per_year=_none(r.segment_expected_severe_per_year)),
        **meta) for r in out.itertuples()]


def _to_frame(scenarios: List[CollisionScenario], feature_set: str) -> pd.DataFrame:
    rows = []
    for i, sc in enumerate(scenarios):
        d = sc.model_dump()
        if feature_set == "claims_triage":
            missing = [f for f in _POST_CRASH if d.get(f) is None]
            if missing:
                raise HTTPException(422, detail=f"scenario {i}: claims_triage model needs "
                                                f"{missing}")
        ts = d.pop("timestamp") or datetime.now(_TZ)
        if ts.tzinfo is not None:                      # training data is LA local time
            ts = ts.astimezone(_TZ).replace(tzinfo=None)
        d["timestamp"] = ts
        d["Latitude"], d["Longitude"] = d.pop("latitude"), d.pop("longitude")
        rows.append(d)
    return pd.DataFrame(rows)


def _none(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    return v
