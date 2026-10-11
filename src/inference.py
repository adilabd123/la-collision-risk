"""
Pieces shared by training and serving, so both build features identically.

* ``HotspotMatcher`` – "DBSCAN predict": assigns a new crash to the training
  hotspot whose *core* point lies within that stage's eps (DBSCAN's own rule for
  border points), checking the junction stage before the corridor stage.
* ``ModelFeatureBuilder`` – picks model columns for the chosen feature set and
  derives ``vehicle_age``.
* ``RiskScorer`` – end-to-end inference: reproject → snap → enrich → match
  hotspot → model → threshold → 4-tier risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from config import settings
from config.config import ModelConfig
from src.clustering import NOISE, ClusteringResult
from src.data_generator import lapd_area_for, lighting_from_time
from src.features import CollinearityScreener, FeatureEngineeringPipeline
from src.ingestion import RoadNetworkLoader, RoadSnapper, SpatialProjector
from src.utils import get_logger

logger = get_logger(__name__)

HOTSPOT_NONE = "none"


# --------------------------------------------------------------------------- #
# Hotspot matching ("DBSCAN predict")
# --------------------------------------------------------------------------- #
@dataclass
class _StageCores:
    name: str
    eps_m: float
    xy: np.ndarray            # core-point coordinates (projected metres)
    labels: np.ndarray        # final cluster id of each core point
    _tree: Optional[cKDTree] = field(default=None, repr=False)

    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.xy)
        return self._tree

    def __getstate__(self) -> dict:          # don't pickle the KD-tree
        d = dict(self.__dict__)
        d["_tree"] = None
        return d


class HotspotMatcher:
    """Assign points to training hotspots using each stage's core points and eps."""

    def __init__(self, stages: List[_StageCores], clusters: pd.DataFrame) -> None:
        self.stages = stages
        # cluster_id -> type / size / loss rank
        self.clusters = clusters.set_index("cluster_id")[
            ["cluster_type", "n_collisions", "loss_rank"]]

    @classmethod
    def from_result(cls, result: ClusteringResult) -> "HotspotMatcher":
        """Recover DBSCAN core points per stage from a fitted clustering result."""
        d = result.data
        X = np.column_stack([d.geometry.x.to_numpy(), d.geometry.y.to_numpy()])
        labels = d["cluster_id"].to_numpy()
        stage_of = d["cluster_stage"].to_numpy()
        stages, pool = [], np.ones(len(d), dtype=bool)        # stage 1 saw every point
        for r in result.stages:
            Xs = X[pool]
            counts = NearestNeighbors(radius=r.eps_m).fit(Xs) \
                .radius_neighbors(Xs, return_distance=False)
            n_nb = np.fromiter((len(c) for c in counts), int, len(Xs))   # includes self
            member = (stage_of[pool] == r.stage) & (labels[pool] != NOISE)
            core = member & (n_nb >= r.min_samples)
            stages.append(_StageCores(r.stage, r.eps_m, Xs[core], labels[pool][core]))
            pool &= stage_of != r.stage                          # next stage sees the rest
        logger.info("Matcher: %s core points",
                    {s.name: len(s.xy) for s in stages})
        return cls(stages, pd.DataFrame(result.clusters.drop(columns="geometry")))

    def match(self, X: np.ndarray) -> np.ndarray:
        """Cluster id per point, or NOISE (-1)."""
        out = np.full(len(X), NOISE)
        for s in self.stages:
            todo = out == NOISE
            if not todo.any() or len(s.xy) == 0:
                continue
            dist, idx = s.tree().query(X[todo], k=1, distance_upper_bound=s.eps_m)
            hit = np.isfinite(dist)
            sub = np.flatnonzero(todo)[hit]
            out[sub] = s.labels[idx[hit]]
        return out

    def describe(self, labels: np.ndarray) -> pd.DataFrame:
        """Label-free hotspot features for model input and API responses."""
        info = self.clusters.reindex(labels)
        in_hot = labels != NOISE
        return pd.DataFrame({
            "cluster_id": labels,
            "in_hotspot": in_hot.astype(int),
            "hotspot_type": np.where(in_hot, info["cluster_type"].fillna(HOTSPOT_NONE),
                                     HOTSPOT_NONE),
            "hotspot_log_size": np.log1p(info["n_collisions"].fillna(0).to_numpy()),
            "hotspot_loss_rank": info["loss_rank"].to_numpy(),
        })


# --------------------------------------------------------------------------- #
# Model column selection
# --------------------------------------------------------------------------- #
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived inputs used by every feature set."""
    df = df.copy()
    if "vehicle_year" in df:
        year = pd.to_datetime(df["timestamp"]).dt.year
        df["vehicle_age"] = (year - pd.to_numeric(df["vehicle_year"], errors="coerce")) \
            .clip(lower=0)
    return df


class ModelFeatureBuilder:
    """Choose numeric + categorical model inputs for a feature set; screen on train only."""

    def __init__(self, cfg: ModelConfig = settings.model, feature_set: Optional[str] = None):
        self.cfg = cfg
        self.feature_set = feature_set or cfg.feature_set
        if self.feature_set not in ("underwriting", "claims_triage"):
            raise ValueError(f"Unknown feature_set {self.feature_set!r}")
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.screening = None

    def _blocked(self) -> set:
        c = self.cfg
        blocked = set(c.non_feature_columns) | set(c.protected_attributes) | set(c.excluded_numeric)
        if self.feature_set == "underwriting":
            blocked |= set(c.post_crash_columns)
        return blocked

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "ModelFeatureBuilder":
        blocked = self._blocked()
        self.categorical_cols = [c for c in self.cfg.categorical_columns + ["area_name"]
                                 if c in df and c not in blocked]
        numeric = [c for c in df.columns
                   if c not in blocked and c not in self.categorical_cols
                   and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
                   and df[c].notna().any()]
        protected_cyc = [c for c in numeric if "_Sin" in c or "_Cos" in c]
        # driver_age is protected so its near-duplicate license_years is the one
        # dropped: age is the rating factor an API caller actually supplies.
        report = CollinearityScreener(
            settings.features,
            protected=protected_cyc + ["Collision_Hour", "DayOfWeek", "Month", "driver_age"],
        ).fit(df, numeric, target=y)
        self.numeric_cols, self.screening = report.kept, report
        logger.info("Feature set '%s': %d numeric, %d categorical (dropped %s)",
                    self.feature_set, len(self.numeric_cols), len(self.categorical_cols),
                    report.dropped)
        return self

    @property
    def input_columns(self) -> List[str]:
        """Columns the sklearn pipeline expects (cluster_id feeds the target encoder)."""
        return self.numeric_cols + self.categorical_cols + ["cluster_id"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.numeric_cols:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float) if c in df else np.nan
        for c in self.categorical_cols:
            out[c] = df[c].astype("string").fillna("missing").astype(object) if c in df \
                else "missing"
        out["cluster_id"] = df["cluster_id"].astype(np.int64)
        return out


# --------------------------------------------------------------------------- #
# Risk tiers
# --------------------------------------------------------------------------- #
def assign_tier(p: np.ndarray, cutoffs: Sequence[float],
                labels: Sequence[str] = settings.model.tier_labels) -> np.ndarray:
    """Map probabilities to ordered tier labels using ascending cut points."""
    return np.asarray(labels, dtype=object)[np.searchsorted(np.asarray(cutoffs), p, side="right")]


# --------------------------------------------------------------------------- #
# End-to-end scorer
# --------------------------------------------------------------------------- #
class OffNetworkError(ValueError):
    """Raised when a location is too far from any road to be scored."""


class RiskScorer:
    """
    Load a trained bundle plus the road graph and score raw crash scenarios.

    The bundle (written by ``train.py``) holds the fitted sklearn/imblearn
    pipeline, feature builder, hotspot matcher, threshold, tier cut points and
    the segment risk lookup.
    """

    def __init__(self, bundle: dict, G: nx.MultiDiGraph) -> None:
        self.bundle = bundle
        self.model = bundle["model"]
        self.builder: ModelFeatureBuilder = bundle["builder"]
        self.matcher: HotspotMatcher = bundle["matcher"]
        self.threshold: float = bundle["threshold"]
        self.cutoffs: List[float] = bundle["tier_cutoffs"]
        self.segment_tiers: Dict[Tuple[int, int], dict] = bundle.get("segment_tiers", {})
        loader = RoadNetworkLoader()
        self.G = loader.project(G)
        self.projector = SpatialProjector()
        self.snapper = RoadSnapper(self.G)
        self.features = FeatureEngineeringPipeline(self.G)
        self._check_graph()

    @classmethod
    def load(cls, model_path: Optional[Path] = None,
             graphml_path: Optional[Path] = None) -> "RiskScorer":
        model_path = Path(model_path or settings.api.model_path or settings.paths.model_bundle)
        bundle = joblib.load(model_path)
        G = RoadNetworkLoader().load(Path(graphml_path or settings.api.graphml_path
                                          or settings.paths.road_graph))
        return cls(bundle, G)

    def _check_graph(self) -> None:
        fp = self.bundle.get("graph_fingerprint", {})
        if fp and fp.get("n_edges") != self.G.number_of_edges():
            logger.warning("Road graph differs from training graph (%s vs %s edges); hotspot "
                           "and segment lookups may not line up.",
                           fp.get("n_edges"), self.G.number_of_edges())

    @staticmethod
    def _fill_context(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Derive inputs a caller doesn't send, exactly as the generator did:
        lighting from sun position (street lights assumed), LAPD area from
        location, road surface from weather.
        """
        gdf = gdf.copy()
        if "lighting" not in gdf or gdf["lighting"].isna().any():
            derived = lighting_from_time(gdf["timestamp"], street_lights=True)
            gdf["lighting"] = gdf["lighting"].fillna(pd.Series(derived, index=gdf.index)) \
                if "lighting" in gdf else derived
        gdf["area_name"] = lapd_area_for(gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy())
        if "weather" not in gdf:
            gdf["weather"] = "clear"
        if "road_surface" not in gdf or gdf["road_surface"].isna().any():
            derived = np.where(gdf["weather"].astype(str) == "rain", "wet", "dry")
            gdf["road_surface"] = gdf["road_surface"].fillna(pd.Series(derived, index=gdf.index)) \
                if "road_surface" in gdf else derived
        return gdf

    def score(self, records: pd.DataFrame, off_network: str = "raise") -> pd.DataFrame:
        """
        Score one row per scenario. Required: Latitude, Longitude, timestamp.

        Rows more than ``reject_snap_distance_m`` from any road either raise
        ``OffNetworkError`` (``off_network="raise"``) or come back with an
        ``error`` message and no score (``"flag"``, used for batches).
        """
        df = records.reset_index(drop=True).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        snapped = self.snapper.snap(self.projector.to_projected(df))
        limit = settings.ingestion.reject_snap_distance_m
        far = (snapped["snap_distance_m"] > limit).to_numpy()
        msg = f"location is more than {limit:.0f} m from any road"
        if far.any() and off_network == "raise":
            raise OffNetworkError(f"{int(far.sum())} {msg}: rows {np.flatnonzero(far).tolist()}")
        if far.all():
            return pd.DataFrame({"error": [msg] * len(df)})
        ok = self._score_snapped(snapped.loc[~far])
        ok.index = np.flatnonzero(~far)
        out = ok.reindex(range(len(df)))            # off-network rows become empty
        out["error"] = np.where(far, msg, None)
        return out

    def _score_snapped(self, snapped: gpd.GeoDataFrame) -> pd.DataFrame:
        snapped = snapped.reset_index(drop=True)

        snapped = self._fill_context(snapped)
        enriched, _ = self.features.enrich(snapped)
        X = np.column_stack([enriched.geometry.x, enriched.geometry.y])
        hot = self.matcher.describe(self.matcher.match(X))
        enriched = pd.concat([enriched.drop(columns=[c for c in hot.columns if c in enriched]),
                              hot.set_index(enriched.index)], axis=1)
        enriched = add_derived(enriched)

        proba = self.model.predict_proba(self.builder.transform(enriched))[:, 1]
        seg = [self.segment_tiers.get((min(u, v), max(u, v)), {})
               for u, v in zip(enriched["edge_u"], enriched["edge_v"])]
        snapped_geo = gpd.GeoSeries(enriched.geometry, crs=enriched.crs).to_crs(4326)
        return pd.DataFrame({
            "probability_high_risk": proba,
            "high_risk": proba >= self.threshold,
            "risk_tier": assign_tier(proba, self.cutoffs),
            "snapped_latitude": snapped_geo.y.to_numpy(),
            "snapped_longitude": snapped_geo.x.to_numpy(),
            "snap_distance_m": enriched["snap_distance_m"].to_numpy(),
            "snap_quality": enriched["snap_quality"].to_numpy(),
            "road_name": enriched.get("road_name", pd.Series(pd.NA, index=enriched.index)).to_numpy(),
            "road_class": enriched.get("road_class", pd.Series(pd.NA, index=enriched.index)).to_numpy(),
            "hotspot_id": hot["cluster_id"].to_numpy(),
            "hotspot_type": hot["hotspot_type"].to_numpy(),
            "hotspot_loss_rank": hot["hotspot_loss_rank"].to_numpy(),
            "segment_risk_tier": [s.get("tier") for s in seg],
            "segment_expected_severe_per_year": [s.get("expected_severe_per_year") for s in seg],
        })
