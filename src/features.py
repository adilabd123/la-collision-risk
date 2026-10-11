"""
Feature engineering: temporal encodings, simulated API enrichment, collinearity screening.

Input is the road-snapped GeoParquet from ``ingestion.py`` (EPSG:32611). Output is
one feature table per collision plus a screening report.

Layers
------
* ``TemporalFeatureExtractor`` – ``Collision_Hour``, ``DayOfWeek``, ``Month`` and
  Fourier (sin/cos) encodings. Hour 23 and hour 0 are neighbours on a clock but
  23 apart as integers; sin/cos puts them next to each other. The 12-hour
  harmonic lets a linear model represent the two daily rush-hour peaks.
* Enrichment providers – each mimics an external API behind a common
  ``EnrichmentProvider`` interface, so a real client (Open-Meteo, TomTom/HERE,
  OSM/Google Places) can replace the simulator without touching the pipeline:

  - ``SimulatedWeatherProvider`` – precipitation, visibility, surface friction
    (with LA's "first rain" oil-slick effect after dry spells).
  - ``SimulatedTrafficProvider`` – segment congestion index, observed speed,
    speed differential vs free flow and speed turbulence. Values are a
    deterministic hash of (segment, date, hour), so a segment queried twice
    for the same hour returns the same answer, as a real API would.
  - ``POIDensityProvider`` – alcohol-serving venues within 200 m (simulated or
    OSM-sourced).
  - ``IntersectionComplexityProvider`` – legs, road-class mix, distance to the
    nearest junction and junction count nearby, from the road graph.
* ``CollinearityScreener`` – Pearson correlation matrix, greedy removal of one
  feature from every pair above ``|r| > 0.85``, then VIF on what remains.

Usage::

    python -m src.features
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import Polygon

from config import settings
from config.config import CRSConfig, FeatureConfig, PathConfig, StudyAreaConfig
from src.graph_utils import ROAD_CLASS_LOAD, ROAD_CLASS_SPEED_MPH, intersections_gdf
from src.utils import get_logger, timed

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Deterministic hashing (stable "API" responses for the same key)
# --------------------------------------------------------------------------- #
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
    return x ^ (x >> np.uint64(31))


def hash_uniform(seed: int, *keys: np.ndarray) -> np.ndarray:
    """Map integer key tuples to U(0, 1) deterministically (vectorised)."""
    with np.errstate(over="ignore"):
        h = np.full(len(keys[0]), np.uint64(seed), dtype=np.uint64)
        for k in keys:
            h = _splitmix64(h ^ np.asarray(k).astype(np.int64).view(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def hash_normal(seed: int, *keys: np.ndarray) -> np.ndarray:
    """Deterministic standard-normal draws via Box–Muller on two hashed uniforms."""
    u1 = np.clip(hash_uniform(seed, *keys), 1e-12, 1.0)
    u2 = hash_uniform(seed + 1, *keys)
    return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)


# --------------------------------------------------------------------------- #
# 1. Temporal features
# --------------------------------------------------------------------------- #
class TemporalFeatureExtractor:
    """Calendar fields plus cyclical (Fourier) encodings of hour, weekday and month."""

    def __init__(self, cfg: FeatureConfig = settings.features) -> None:
        self.cfg = cfg

    def transform(self, df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
        ts = pd.to_datetime(df[ts_col])
        hour = ts.dt.hour
        hour_frac = hour + ts.dt.minute / 60.0
        dow = ts.dt.dayofweek
        out = pd.DataFrame(index=df.index)
        out["Collision_Hour"] = hour.astype(int)               # 0–23
        out["Collision_Hour_Frac"] = hour_frac.round(4)        # continuous 0–23.98
        out["DayOfWeek"] = dow.astype(int)                     # 0 = Monday
        out["Month"] = ts.dt.month.astype(int)
        out["IsWeekend"] = (dow >= 5).astype(int)
        out["IsRushHour"] = (hour.isin(self.cfg.rush_hours) & (dow < 5)).astype(int)
        out["IsNight"] = hour.isin(self.cfg.night_hours).astype(int)
        out["WeekendNight"] = out["IsWeekend"] * out["IsNight"]

        # Diurnal harmonics: k=1 → 24 h cycle, k=2 → 12 h cycle (bimodal rush hours)
        for k in range(1, self.cfg.diurnal_harmonics + 1):
            angle = 2 * np.pi * k * hour_frac / 24.0
            out[f"Hour_Sin_{24 // k}h"] = np.sin(angle)
            out[f"Hour_Cos_{24 // k}h"] = np.cos(angle)
        out["DoW_Sin"] = np.sin(2 * np.pi * dow / 7)
        out["DoW_Cos"] = np.cos(2 * np.pi * dow / 7)
        out["Month_Sin"] = np.sin(2 * np.pi * (ts.dt.month - 1) / 12)
        out["Month_Cos"] = np.cos(2 * np.pi * (ts.dt.month - 1) / 12)
        return out


# --------------------------------------------------------------------------- #
# 2. Enrichment providers
# --------------------------------------------------------------------------- #
class EnrichmentProvider(ABC):
    """Interface for an external data layer joined onto each collision."""

    name: str = "provider"

    @abstractmethod
    def enrich(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Return new columns aligned to ``gdf.index``."""


class SimulatedWeatherProvider(EnrichmentProvider):
    """
    Historical weather at crash time (stands in for e.g. Open-Meteo's archive API).

    Uses the record's own ``weather`` / ``road_surface`` label where present,
    as if the API confirmed the police report, and adds continuous measures.
    ``days_since_rain`` comes from a simulated daily rain calendar (Markov
    chain with LA's winter wet season).
    """

    name = "weather"

    def __init__(self, cfg: FeatureConfig = settings.features) -> None:
        self.seed = cfg.enrichment_seed

    def enrich(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(gdf["timestamp"])
        n = len(gdf)
        key = ts.astype("int64").to_numpy() // 3_600_000_000_000  # hour bucket
        # Spatial key: road segment if snapped, else ~100 m grid cell. Keyed on
        # place + hour (never row position) so the same crash gets the same
        # weather in a training batch and in a single API request.
        if "edge_u" in gdf:
            rid = np.minimum(gdf["edge_u"], gdf["edge_v"]).to_numpy(np.int64)
        else:
            rid = (np.floor(gdf.geometry.x.to_numpy() / 100).astype(np.int64) * 100_003
                   + np.floor(gdf.geometry.y.to_numpy() / 100).astype(np.int64))
        label = gdf["weather"].astype(str).to_numpy() if "weather" in gdf \
            else np.full(n, "clear")
        u = hash_uniform(self.seed, key, rid)

        precip = np.where(label == "rain",
                          np.clip(-np.log(np.clip(u, 1e-9, 1)) * 2.2, 0.2, 25.0),   # mm/h
                          np.where(label == "cloudy", np.where(u < 0.08, 0.1, 0.0), 0.0))

        base_vis = {"clear": 16.0, "cloudy": 14.0, "rain": 9.0, "fog": 0.9}
        vis = np.array([base_vis.get(w, 16.0) for w in label])
        vis = vis * np.exp(-0.06 * precip) * (0.85 + 0.3 * hash_uniform(self.seed + 7, key, rid))
        night = ts.dt.hour.isin(settings.features.night_hours).to_numpy()
        vis = np.where(night & (label == "fog"), vis * 0.6, vis)

        since = self._rain_calendar(ts)
        day = ts.dt.normalize()
        dsr_prev = since.reindex(day - pd.Timedelta(days=1)).fillna(365).to_numpy()
        dsr = np.where(label == "rain", 0, since.reindex(day).fillna(365).to_numpy())
        wet = (gdf["road_surface"].astype(str).to_numpy() == "wet") if "road_surface" in gdf \
            else precip > 0
        # Friction μ: dry asphalt ≈ 0.80, wet ≈ 0.55; heavier rain lowers it;
        # first rain after a long dry spell lifts oil to the surface ("first flush").
        first_flush = np.where(wet, 0.10 * np.clip(dsr_prev / 30.0, 0, 1), 0.0)
        mu = np.where(wet, 0.55, 0.80) - 0.025 * np.minimum(precip, 6.0) - first_flush
        mu = mu + 0.02 * hash_normal(self.seed + 3, key, rid)

        return pd.DataFrame({
            "precip_mm_hr": np.round(precip, 2),
            "visibility_km": np.round(np.clip(vis, 0.05, 16.0), 2),
            "days_since_rain": dsr.astype(int),
            "dry_spell_before_rain": np.where(wet, dsr_prev, 0).astype(int),
            "surface_friction_index": np.round(np.clip(mu, 0.25, 0.85), 3),
        }, index=gdf.index)

    CALENDAR_START = pd.Timestamp("2015-01-01")

    def _rain_calendar(self, ts: pd.Series) -> pd.Series:
        """
        Days since last rain for every calendar day (simulated daily history).

        Anchored at a fixed start date so the calendar is identical whatever
        dates a batch contains — required for train/serve consistency.
        """
        end = max(ts.max().normalize(), pd.Timestamp.today().normalize())
        cached = getattr(self, "_calendar", None)
        if cached is not None and cached.index[-1] >= end:
            return cached
        days = pd.date_range(self.CALENDAR_START, end, freq="D")
        rng = np.random.default_rng(self.seed)
        wet_season = np.isin(days.month, [11, 12, 1, 2, 3])
        rain = np.zeros(len(days), dtype=bool)
        for i in range(1, len(days)):  # 2-state Markov chain: rain persists
            p = (0.45 if rain[i - 1] else 0.12) if wet_season[i] else \
                (0.15 if rain[i - 1] else 0.01)
            rain[i] = rng.random() < p
        idx = np.arange(len(days))
        last = np.maximum.accumulate(np.where(rain, idx, -10_000))
        self._calendar = pd.Series(np.minimum(idx - last, 365), index=days)
        return self._calendar


class SimulatedTrafficProvider(EnrichmentProvider):
    """
    Segment-level traffic at crash time (stands in for TomTom/HERE flow APIs).

    congestion = sigmoid(demand(hour, weekday) × road-class load
                         + persistent segment bottleneck + day-to-day noise)
    observed speed follows a BPR-style drop with congestion; ``speed_std_mph``
    peaks at medium congestion, the stop-and-go regime where rear-end crashes
    cluster.
    """

    name = "traffic"
    _WEEKDAY = np.array([0.15, 0.10, 0.08, 0.08, 0.12, 0.30, 0.65, 0.95, 1.00, 0.75, 0.65,
                         0.68, 0.72, 0.72, 0.78, 0.90, 1.00, 1.00, 0.85, 0.62, 0.48, 0.40,
                         0.32, 0.22])
    _WEEKEND = np.array([0.25, 0.18, 0.14, 0.10, 0.08, 0.10, 0.18, 0.30, 0.45, 0.58, 0.66,
                         0.72, 0.75, 0.75, 0.74, 0.72, 0.70, 0.66, 0.60, 0.52, 0.45, 0.40,
                         0.36, 0.30])

    def __init__(self, cfg: FeatureConfig = settings.features) -> None:
        self.seed = cfg.enrichment_seed + 101

    def enrich(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(gdf["timestamp"])
        hour = ts.dt.hour.to_numpy()
        weekend = ts.dt.dayofweek.to_numpy() >= 5
        day = (ts.dt.normalize().astype("int64") // 86_400_000_000_000).to_numpy()
        u = gdf["edge_u"].to_numpy(np.int64)
        v = gdf["edge_v"].to_numpy(np.int64)
        seg = np.minimum(u, v) * 1_000_003 + np.maximum(u, v)   # direction-agnostic id

        cls = gdf.get("road_class", pd.Series("residential", index=gdf.index)).astype(str)
        load = cls.map(ROAD_CLASS_LOAD).fillna(1.0).to_numpy()
        free_flow = gdf.get("speed_limit_mph", pd.Series(np.nan, index=gdf.index)).astype(float)
        free_flow = free_flow.fillna(cls.map(ROAD_CLASS_SPEED_MPH)).fillna(25.0).to_numpy()

        demand = np.where(weekend, self._WEEKEND[hour], self._WEEKDAY[hour])
        bottleneck = 0.8 * hash_normal(self.seed, seg)               # persistent
        daily = 0.5 * hash_normal(self.seed + 1, seg, day, hour)      # varies by day/hour
        z = 4.0 * demand * np.log1p(load) - 2.6 + bottleneck + daily
        congestion = 1.0 / (1.0 + np.exp(-z))

        observed = free_flow * (1.0 - 0.72 * congestion ** 1.5)
        turbulence = 4.0 * congestion * (1 - congestion)              # 0..1, peak at 0.5
        speed_std = free_flow * (0.05 + 0.20 * turbulence)
        return pd.DataFrame({
            "congestion_index": np.round(congestion, 4),
            "free_flow_speed_mph": free_flow,
            "observed_speed_mph": np.round(observed, 2),
            "speed_differential_mph": np.round(free_flow - observed, 2),
            "speed_ratio": np.round(observed / free_flow, 4),
            "speed_std_mph": np.round(speed_std, 2),
        }, index=gdf.index)


# ---- venue sources for POI density ----------------------------------------- #
class VenueSource(ABC):
    """Supplies alcohol-serving venue points in the projected CRS."""

    @abstractmethod
    def load(self) -> gpd.GeoSeries:
        ...


class SimulatedVenueSource(VenueSource):
    """
    Synthetic bars / liquor stores clustered in LA nightlife districts plus a
    commercial-strip background. Cached to GeoParquet.
    """

    NIGHTLIFE = {  # (lat, lon, relative weight, spread m)
        "Hollywood": (34.1010, -118.3310, 3.0, 700), "Downtown": (34.0450, -118.2550, 2.6, 800),
        "Koreatown": (34.0610, -118.3000, 2.2, 650), "Venice": (33.9900, -118.4650, 1.4, 500),
        "Silver Lake": (34.0870, -118.2700, 1.2, 600), "Westwood": (34.0630, -118.4460, 0.9, 500),
        "NoHo": (34.1650, -118.3770, 0.9, 550), "Highland Park": (34.1120, -118.1930, 0.8, 500),
        "Sherman Oaks": (34.1510, -118.4490, 0.8, 700), "San Pedro": (33.7380, -118.2850, 0.5, 450),
    }

    def __init__(self, cfg: FeatureConfig = settings.features,
                 area_cfg: StudyAreaConfig = settings.study_area,
                 crs_cfg: CRSConfig = settings.crs,
                 cache: Optional[Path] = settings.paths.venues_cache) -> None:
        self.n, self.seed, self.cache = cfg.n_simulated_venues, cfg.enrichment_seed + 202, cache
        self.area_cfg, self.crs = area_cfg, crs_cfg

    def load(self) -> gpd.GeoSeries:
        if self.cache and self.cache.exists():
            return gpd.read_parquet(self.cache).geometry
        rng = np.random.default_rng(self.seed)
        to_utm = Transformer.from_crs(self.crs.geographic, self.crs.projected, always_xy=True)
        names = list(self.NIGHTLIFE)
        w = np.array([self.NIGHTLIFE[k][2] for k in names])
        n_hot = int(self.n * 0.7)
        pick = rng.choice(len(names), n_hot, p=w / w.sum())
        cx, cy = to_utm.transform([self.NIGHTLIFE[k][1] for k in names],
                                  [self.NIGHTLIFE[k][0] for k in names])
        sd = np.array([self.NIGHTLIFE[k][3] for k in names])[pick]
        hx = np.asarray(cx)[pick] + rng.normal(0, 1, n_hot) * sd
        hy = np.asarray(cy)[pick] + rng.normal(0, 1, n_hot) * sd

        land = Polygon(self.area_cfg.land_polygon)
        bx, by = [], []
        w_, s_, e_, n_ = self.area_cfg.bbox
        while len(bx) < self.n - n_hot:
            lon, lat = rng.uniform(w_, e_, 2000), rng.uniform(s_, n_, 2000)
            keep = shapely.contains_xy(land, lon, lat)
            x, y = to_utm.transform(lon[keep], lat[keep])
            bx.extend(x), by.extend(y)
        x = np.concatenate([hx, bx[: self.n - n_hot]])
        y = np.concatenate([hy, by[: self.n - n_hot]])
        venues = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=self.crs.projected)
        if self.cache:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            gpd.GeoDataFrame(geometry=venues).to_parquet(self.cache)
        return venues


class OSMVenueSource(VenueSource):
    """Real alcohol-serving venues from OpenStreetMap tags (bars, pubs, liquor shops)."""

    def __init__(self, cfg: FeatureConfig = settings.features,
                 area_cfg: StudyAreaConfig = settings.study_area,
                 crs_cfg: CRSConfig = settings.crs,
                 cache: Optional[Path] = settings.paths.venues_cache) -> None:
        self.tags, self.bbox = cfg.osm_alcohol_tags, area_cfg.bbox
        self.crs, self.cache = crs_cfg.projected, cache

    def load(self) -> gpd.GeoSeries:
        if self.cache and self.cache.exists():
            return gpd.read_parquet(self.cache).geometry
        with timed(logger, "OSM venue download"):
            feats = ox.features_from_bbox(self.bbox, tags=self.tags)
        pts = feats.to_crs(self.crs).geometry.representative_point().reset_index(drop=True)
        if self.cache:
            gpd.GeoDataFrame(geometry=pts).to_parquet(self.cache)
        logger.info("Loaded %d OSM alcohol venues", len(pts))
        return pts


class POIDensityProvider(EnrichmentProvider):
    """Count of alcohol-serving venues within ``poi_buffer_m`` and distance to the nearest."""

    name = "poi"

    def __init__(self, source: VenueSource, cfg: FeatureConfig = settings.features) -> None:
        self.source, self.radius = source, cfg.poi_buffer_m
        self._tree: Optional[shapely.STRtree] = None    # built once, reused per request

    def enrich(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        if self._tree is None:
            venues = self.source.load()
            if venues.crs is not None and not venues.crs.equals(gdf.crs):
                venues = venues.to_crs(gdf.crs)
            self._tree = shapely.STRtree(venues.to_numpy())
        tree = self._tree
        pts = gdf.geometry.to_numpy()
        hits = tree.query(pts, predicate="dwithin", distance=self.radius)
        counts = np.bincount(hits[0], minlength=len(gdf))
        _, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
        r = int(self.radius)
        return pd.DataFrame({f"alcohol_venues_{r}m": counts,
                             "dist_nearest_alcohol_venue_m": np.round(dist, 1)},
                            index=gdf.index)


class IntersectionComplexityProvider(EnrichmentProvider):
    """
    Junction geometry around each crash, from the projected road graph.

    ``intersection_complexity`` = ((legs − 2) + 0.5·(class_mix − 1)) · exp(−d / 40 m):
    a 5-leg junction where an arterial meets residential streets scores high
    when the crash is on it, and the score fades with distance.
    """

    name = "intersections"

    def __init__(self, G: nx.MultiDiGraph, cfg: FeatureConfig = settings.features) -> None:
        self.cfg = cfg
        self.junctions = intersections_gdf(G, cfg.intersection_min_legs)
        self._tree = shapely.STRtree(self.junctions.geometry.to_numpy())

    def enrich(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        j, tree = self.junctions, self._tree
        pts = gdf.geometry.to_numpy()
        (_, near), dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
        legs = j["legs"].to_numpy()[near]
        mix = j["class_mix"].to_numpy()[near]
        within = np.bincount(tree.query(pts, predicate="dwithin", distance=200.0)[0],
                             minlength=len(gdf))
        score = ((legs - 2) + 0.5 * (mix - 1)) * np.exp(-dist / self.cfg.intersection_decay_m)
        return pd.DataFrame({
            "dist_to_intersection_m": np.round(dist, 1),
            "intersection_legs": legs,
            "intersection_class_mix": mix,
            "at_intersection": (dist <= self.cfg.at_intersection_m).astype(int),
            "intersections_within_200m": within,
            "intersection_complexity": np.round(score, 4),
        }, index=gdf.index)


# --------------------------------------------------------------------------- #
# 3. Collinearity screening
# --------------------------------------------------------------------------- #
@dataclass
class ScreeningReport:
    """Result of collinearity screening."""

    corr: pd.DataFrame
    high_pairs: pd.DataFrame
    dropped: List[str]
    kept: List[str]
    vif: pd.Series = field(default_factory=pd.Series)
    threshold: float = 0.85

    def summary(self) -> str:
        top_vif = self.vif.sort_values(ascending=False).head(3).round(2).to_dict()
        return (f"{len(self.kept)} kept, {len(self.dropped)} dropped {self.dropped}; "
                f"{len(self.high_pairs)} pairs above threshold; top VIF {top_vif}")


class CollinearityScreener:
    """
    Remove redundant numeric features using a Pearson correlation matrix.

    For each pair with ``|r| > corr_threshold`` (strongest first) one member is
    dropped: the one less correlated with ``target`` if given, otherwise the
    one with the higher mean ``|r|`` to all other features (more redundant).
    ``protected`` features are never dropped — e.g. sin/cos pairs, which only
    work together. VIF (diagonal of the inverse correlation matrix) is
    reported on the survivors as a multivariate check.
    """

    def __init__(self, cfg: FeatureConfig = settings.features,
                 protected: Iterable[str] = ()) -> None:
        self.threshold, self.vif_threshold = cfg.corr_threshold, cfg.vif_threshold
        self.protected = set(protected)

    def fit(self, df: pd.DataFrame, features: Sequence[str],
            target: Optional[pd.Series] = None) -> ScreeningReport:
        X = df[list(features)].astype(float)
        X = X.loc[:, X.std(ddof=0) > 0]                    # constant columns carry no info
        corr = X.corr(method="pearson")
        abs_c = corr.abs().where(~np.eye(len(corr), dtype=bool))
        mean_r = abs_c.mean()
        tgt_r = X.corrwith(target.astype(float)).abs() if target is not None else None

        iu = np.triu_indices(len(corr), k=1)
        pairs = pd.DataFrame({"feature_a": corr.index[iu[0]], "feature_b": corr.columns[iu[1]],
                              "r": corr.to_numpy()[iu]})
        pairs = pairs[pairs["r"].abs() > self.threshold] \
            .sort_values("r", key=np.abs, ascending=False).reset_index(drop=True)

        dropped: List[str] = []
        for a, b in pairs[["feature_a", "feature_b"]].itertuples(index=False):
            if a in dropped or b in dropped:
                continue
            cand = [f for f in (a, b) if f not in self.protected]
            if not cand:
                continue
            if len(cand) == 1:
                victim = cand[0]
            elif tgt_r is not None:
                victim = min(cand, key=lambda f: tgt_r[f])
            else:
                victim = max(cand, key=lambda f: mean_r[f])
            dropped.append(victim)

        kept = [c for c in X.columns if c not in dropped]
        vif = self._vif(X[kept])
        flagged = vif[vif > self.vif_threshold]
        if len(flagged):
            logger.warning("VIF above %.0f after screening: %s", self.vif_threshold,
                           flagged.round(1).to_dict())
        return ScreeningReport(corr, pairs, dropped, kept, vif, self.threshold)

    @staticmethod
    def _vif(X: pd.DataFrame) -> pd.Series:
        if X.shape[1] < 2:
            return pd.Series(1.0, index=X.columns, name="vif")
        r = np.corrcoef(X.to_numpy(), rowvar=False)
        return pd.Series(np.diag(np.linalg.pinv(r)), index=X.columns, name="vif")

    @staticmethod
    def save_heatmap(report: ScreeningReport, path: Path) -> Optional[Path]:
        """Write a correlation heatmap PNG (requires matplotlib)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:                               # pragma: no cover
            logger.warning("matplotlib not installed; skipping heatmap")
            return None
        c = report.corr
        fig, ax = plt.subplots(figsize=(0.32 * len(c) + 3, 0.32 * len(c) + 2))
        im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(c)), c.index, fontsize=7)
        for i, name in enumerate(c.index):
            if name in report.dropped:
                ax.get_yticklabels()[i].set_color("#b42318")
        fig.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
        ax.set_title(f"Pearson correlation — pairs with |r| > {report.threshold:.2f} screened; "
                     f"{len(report.dropped)} dropped (red)", fontsize=9)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSet:
    """Engineered features plus the list of model-ready columns."""

    data: gpd.GeoDataFrame
    feature_columns: List[str]
    report: ScreeningReport


class FeatureEngineeringPipeline:
    """Temporal features → enrichment layers → collinearity screening → persist."""

    # Context columns from upstream that are useful model inputs as-is.
    PASSTHROUGH_NUMERIC = ("driver_age", "license_years", "prior_claims_3yr", "vehicle_year",
                           "num_vehicles", "speed_limit_mph", "lanes", "edge_length_m",
                           "snap_distance_m")
    PASSTHROUGH_BOOL = ("speeding", "alcohol_involved", "distracted", "hit_and_run", "oneway")

    def __init__(self, G: nx.MultiDiGraph,
                 providers: Optional[Sequence[EnrichmentProvider]] = None,
                 cfg: FeatureConfig = settings.features,
                 paths: PathConfig = settings.paths) -> None:
        self.cfg, self.paths = cfg, paths
        self.temporal = TemporalFeatureExtractor(cfg)
        source = OSMVenueSource(cfg) if cfg.venue_source == "osm" else SimulatedVenueSource(cfg)
        self.providers = list(providers) if providers is not None else [
            SimulatedWeatherProvider(cfg), SimulatedTrafficProvider(cfg),
            POIDensityProvider(source, cfg), IntersectionComplexityProvider(G, cfg)]
        # Never dropped: sin/cos pairs (only meaningful together) and the
        # calendar fields the model spec requires by name.
        protected = [f"Hour_{t}_{24 // k}h" for k in range(1, cfg.diurnal_harmonics + 1)
                     for t in ("Sin", "Cos")] + ["DoW_Sin", "DoW_Cos", "Month_Sin", "Month_Cos",
                                                  "Collision_Hour", "DayOfWeek", "Month"]
        self.screener = CollinearityScreener(cfg, protected=protected)

    def enrich(self, gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, List[pd.DataFrame]]:
        """
        Add temporal and enrichment columns to snapped points (no screening).

        This is the single code path shared by training and the API, so
        features are computed identically in both.
        """
        parts = [self.temporal.transform(gdf)]
        for p in self.providers:
            with timed(logger, f"  enrichment: {p.name}"):
                parts.append(p.enrich(gdf))
        data = gpd.GeoDataFrame(pd.concat([gdf, *parts], axis=1), geometry="geometry",
                                crs=gdf.crs)
        for c in self.PASSTHROUGH_BOOL:
            if c in data:
                data[c] = data[c].astype("Float64").fillna(0).astype(int)
        return data, parts

    def run(self, gdf: gpd.GeoDataFrame, target_col: Optional[str] = "severity_code",
            write: bool = True) -> FeatureSet:
        with timed(logger, f"Feature engineering on {len(gdf):,} rows"):
            data, parts = self.enrich(gdf)
            candidates = self._numeric_candidates(data, parts)
            target = data[target_col] if target_col and target_col in data else None
            report = self.screener.fit(data, candidates, target)
        logger.info("Collinearity screening: %s", report.summary())

        fs = FeatureSet(data, report.kept, report)
        if write:
            self._write(fs)
        return fs

    def _numeric_candidates(self, data: pd.DataFrame, new_parts: List[pd.DataFrame]) -> List[str]:
        excluded = set(settings.model.non_feature_columns) | set(settings.model.protected_attributes)
        cols = [c for p in new_parts for c in p.columns]
        cols += [c for c in self.PASSTHROUGH_NUMERIC + self.PASSTHROUGH_BOOL if c in data]
        return [c for c in dict.fromkeys(cols)
                if c not in excluded and pd.api.types.is_numeric_dtype(data[c])]

    def _write(self, fs: FeatureSet) -> None:
        self.paths.ensure()
        out = fs.data.drop(columns=[c for c in ("raw_geometry",) if c in fs.data])
        out.to_parquet(self.paths.features, index=False)
        rd = self.paths.reports_dir
        fs.report.corr.round(4).to_csv(rd / "feature_correlation_matrix.csv")
        fs.report.high_pairs.to_csv(rd / "collinear_pairs.csv", index=False)
        fs.report.vif.round(3).to_csv(rd / "feature_vif.csv")
        pd.Series(fs.feature_columns, name="feature").to_csv(rd / "selected_features.csv",
                                                              index=False)
        CollinearityScreener.save_heatmap(fs.report, rd / "feature_correlation_heatmap.png")
        logger.info("Wrote features to %s and reports to %s", self.paths.features, rd)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Iterable[str]] = None) -> FeatureSet:
    from src.ingestion import RoadNetworkLoader

    parser = argparse.ArgumentParser(description="Engineer features for LA collisions.")
    parser.add_argument("-i", "--input", type=Path, default=settings.paths.snapped_collisions)
    parser.add_argument("--graphml", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    gdf = gpd.read_parquet(args.input)
    G = RoadNetworkLoader().load(args.graphml)
    return FeatureEngineeringPipeline(G).run(gdf)


if __name__ == "__main__":
    main()
