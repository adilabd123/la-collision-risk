"""Tests for the synthetic collision generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon

from config import settings
from src.data_generator import CollisionDataGenerator

REQUIRED = {"collision_id", "Latitude", "Longitude", "timestamp", "area_name",
            "driver_age", "driver_gender", "vehicle_type", "severity", "claim_amount_usd"}


def test_row_count_and_schema(sample_df: pd.DataFrame) -> None:
    assert len(sample_df) == 5_000
    assert REQUIRED <= set(sample_df.columns)


def test_default_size_is_25k() -> None:
    assert settings.generator.n_records == 25_000


def test_points_on_land_inside_bbox(sample_df: pd.DataFrame) -> None:
    clean = sample_df[(sample_df["Latitude"].notna()) & (sample_df["Latitude"] != 0)]
    land = Polygon(settings.study_area.land_polygon)
    assert shapely.contains_xy(land, clean["Longitude"], clean["Latitude"]).all()


def test_deterministic_for_seed() -> None:
    a = CollisionDataGenerator(seed=7).generate(500)
    b = CollisionDataGenerator(seed=7).generate(500)
    pd.testing.assert_frame_equal(a, b)


def test_dirty_rows_injected(sample_df: pd.DataFrame) -> None:
    assert sample_df["Latitude"].isna().sum() > 0
    assert ((sample_df["Latitude"] == 0) & (sample_df["Longitude"] == 0)).sum() > 0
    assert sample_df["collision_id"].duplicated().sum() > 0


def test_risk_signal_present(sample_df: pd.DataFrame) -> None:
    """Motorcycles and alcohol should raise severity; gender should not (by design)."""
    sev = sample_df["severity_code"]
    moto = sev[sample_df["vehicle_type"] == "motorcycle"].mean()
    sedan = sev[sample_df["vehicle_type"] == "sedan"].mean()
    assert moto > sedan + 0.2
    assert sev[sample_df["alcohol_involved"]].mean() > sev[~sample_df["alcohol_involved"]].mean()
    by_gender = sample_df[sample_df["driver_gender"].isin(["M", "F"])].groupby("driver_gender")
    assert np.ptp(by_gender["severity_code"].mean().to_numpy()) < 0.1


def test_rush_hour_peak(sample_df: pd.DataFrame) -> None:
    wk = sample_df[sample_df["timestamp"].dt.dayofweek < 5]
    hours = wk["timestamp"].dt.hour.value_counts()
    assert hours.loc[16] > 3 * hours.loc[3]


def test_vehicle_not_newer_than_collision(sample_df: pd.DataFrame) -> None:
    assert (sample_df["vehicle_year"] <= sample_df["timestamp"].dt.year + 0).all()
