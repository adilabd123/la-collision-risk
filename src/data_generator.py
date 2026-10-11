"""
Synthetic Los Angeles collision dataset generator.

Produces a realistic 25,000-record dataset whose schema mirrors LAPD open data
(``area_name``, victim/driver demographics, ``Latitude``/``Longitude``) and adds
the insurance fields a risk model needs (``claim_amount_usd``, prior claims,
vehicle attributes).

Realism comes from four built-in structures, each controlled in ``config``:

* **Space** – crashes cluster around LAPD area hotspots (Gaussian mixture in
  metres), with a diffuse background; points are kept on land.
* **Time** – weekday rush-hour peaks, flatter late-night weekend profile,
  winter rain season, and sun-position-aware lighting.
* **Cause → outcome** – severity is an ordinal-logit function of speeding,
  alcohol, motorcycle use, pedestrians, darkness, wet roads and driver age, so
  Step 2 models have genuine signal to learn. ``driver_gender`` has **no**
  causal effect by design (it is a protected attribute).
* **Road network (optional)** – pass an OSMnx graph and a share of crashes is
  planted at complex intersections and along named arterial corridors, with
  heavy-tailed intensities, mirroring how real crashes concentrate. These rows
  carry ``gt_hotspot_id`` / ``gt_hotspot_type`` ground truth so the DBSCAN
  stage can be validated. Without a graph the generator is road-agnostic.
* **Dirt** – a small share of null / (0, 0) coordinates and duplicate IDs, as
  seen in the real LAPD feed, so the ingestion validators are exercised.

Usage::

    python -m src.data_generator                 # writes data/raw/la_collisions_raw.csv
    python -m src.data_generator -n 5000 --seed 7 -o /tmp/sample.csv
    python -m src.data_generator --graphml data/cache/la_drive_network.graphml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon

from config import settings
from config.config import CRSConfig, GeneratorConfig, StudyAreaConfig
from src.graph_utils import (ARTERIAL_CLASSES, ROAD_CLASS_LOAD, edges_gdf,
                             intersections_gdf, walk_named_road)
from src.utils import get_logger, timed

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Shared helpers (also used by the API so inputs are derived identically)
# --------------------------------------------------------------------------- #
def lighting_from_time(ts: pd.Series, street_lights: "np.ndarray | bool" = True) -> np.ndarray:
    """
    Lighting class from approximate LA sunrise/sunset through the year:
    daylight, dusk_dawn (±30 min of sunrise/sunset) or dark (lit / unlit).
    """
    ts = pd.to_datetime(pd.Series(ts))
    doy = ts.dt.dayofyear.to_numpy()
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60
    season = np.cos(2 * np.pi * (doy - 172) / 365.25)
    sunrise, sunset = 6.0 - 0.9 * season, 18.4 + 1.6 * season
    twilight = (np.abs(hour - sunrise) < 0.5) | (np.abs(hour - sunset) < 0.5)
    day = (hour > sunrise) & (hour < sunset) & ~twilight
    lit = np.broadcast_to(np.asarray(street_lights, dtype=bool), day.shape)
    return np.select([day, twilight, lit], ["daylight", "dusk_dawn", "dark_street_lights"],
                     "dark_no_street_lights").astype(object)


def lapd_area_for(x: np.ndarray, y: np.ndarray,
                  area_cfg: StudyAreaConfig = settings.study_area,
                  crs_cfg: CRSConfig = settings.crs) -> np.ndarray:
    """Nearest LAPD area centroid for projected (UTM) coordinates."""
    t = Transformer.from_crs(crs_cfg.geographic, crs_cfg.projected, always_xy=True)
    names = list(area_cfg.lapd_areas)
    cx, cy = t.transform([area_cfg.lapd_areas[k][1] for k in names],
                         [area_cfg.lapd_areas[k][0] for k in names])
    x, y = np.asarray(x, float), np.asarray(y, float)
    d2 = (x[:, None] - np.asarray(cx)[None, :]) ** 2 + (y[:, None] - np.asarray(cy)[None, :]) ** 2
    return np.asarray(names, dtype=object)[d2.argmin(axis=1)]


# --------------------------------------------------------------------------- #
# Static distributions (domain priors, not tuned hyperparameters)
# --------------------------------------------------------------------------- #
_HOUR_WEEKDAY = np.array([0.9, 0.6, 0.5, 0.4, 0.45, 0.8, 1.6, 2.6, 2.8, 2.0, 1.9, 2.1,
                          2.4, 2.4, 2.7, 3.3, 3.5, 3.4, 2.8, 2.2, 1.8, 1.6, 1.4, 1.1])
_HOUR_WEEKEND = np.array([1.9, 1.8, 1.7, 1.2, 0.8, 0.7, 0.8, 1.0, 1.3, 1.6, 1.9, 2.2,
                          2.4, 2.5, 2.5, 2.5, 2.5, 2.4, 2.3, 2.1, 1.9, 1.9, 1.9, 2.0])
_DOW_WEIGHT = np.array([1.00, 1.00, 1.02, 1.05, 1.15, 1.05, 0.90])  # Mon..Sun

_VEHICLE_TYPES: Dict[str, float] = {
    "sedan": 0.44, "suv": 0.27, "pickup": 0.11, "van": 0.06,
    "commercial_truck": 0.05, "motorcycle": 0.04, "bus": 0.01, "bicycle": 0.02,
}
_COLLISION_TYPES: Dict[str, float] = {
    "rear_end": 0.30, "broadside": 0.22, "sideswipe": 0.14, "hit_object": 0.12,
    "pedestrian": 0.08, "head_on": 0.04, "other": 0.10,
}
_INTERSECTION_TYPES: Dict[str, float] = {
    "broadside": 0.42, "pedestrian": 0.14, "rear_end": 0.18, "sideswipe": 0.08,
    "head_on": 0.05, "hit_object": 0.03, "other": 0.10,
}
_CORRIDOR_TYPES: Dict[str, float] = {
    "rear_end": 0.45, "sideswipe": 0.22, "hit_object": 0.10, "broadside": 0.08,
    "head_on": 0.04, "pedestrian": 0.04, "other": 0.07,
}
_SEVERITY_LEVELS = ("property_damage_only", "minor_injury", "severe_injury", "fatal")
_SEVERITY_SHARES = np.array([0.555, 0.330, 0.100, 0.015])
_CLAIM_MEDIAN_USD = np.array([4_200.0, 14_000.0, 65_000.0, 350_000.0])
_CLAIM_SIGMA = np.array([0.60, 0.70, 0.80, 0.90])


@dataclass
class _ProjectedArea:
    """An LAPD area centroid expressed in projected metres."""

    name: str
    x: float
    y: float
    weight: float
    sigma_m: float


class NetworkHotspotPlanter:
    """
    Place crashes on a road graph at a few dangerous intersections and corridors.

    Intersections are chosen with probability ∝ legs × busiest-approach load;
    corridors are walks along a named arterial. Each hotspot gets a
    heavy-tailed (gamma) intensity, so a handful dominate — as in real data —
    while some are too sparse to be detectable.
    """

    def __init__(self, G: nx.MultiDiGraph, cfg: GeneratorConfig, crs_cfg: CRSConfig,
                 rng: np.random.Generator) -> None:
        self.cfg, self.rng = cfg, rng
        if not ox.projection.is_projected(G.graph.get("crs")):
            G = ox.project_graph(G, to_crs=crs_cfg.projected)
        self.G = G
        self.edges = edges_gdf(G)
        self.junctions = intersections_gdf(G)

    def sample(self, n_int: int, n_cor: int) -> pd.DataFrame:
        """Return x/y (projected metres) plus ground-truth ids for planted crashes."""
        return pd.concat([self._intersections(n_int), self._corridors(n_cor)],
                         ignore_index=True)

    def _intensities(self, k: int, n: int) -> np.ndarray:
        w = self.rng.gamma(0.9, 1.0, k)
        return self.rng.multinomial(n, w / w.sum())

    def _intersections(self, n: int) -> pd.DataFrame:
        j = self.junctions
        if n == 0 or j.empty:
            return pd.DataFrame(columns=["x", "y", "gt_hotspot_id", "gt_hotspot_type"])
        p = (j["legs"] * j["max_load"]).to_numpy(float)
        k = min(self.cfg.n_planted_intersections, len(j))
        pick = self.rng.choice(len(j), size=k, replace=False, p=p / p.sum())
        counts = self._intensities(k, n)
        cx = np.repeat(j.geometry.x.to_numpy()[pick], counts)
        cy = np.repeat(j.geometry.y.to_numpy()[pick], counts)
        s = self.cfg.intersection_spread_m
        return pd.DataFrame({
            "x": cx + self.rng.normal(0, s, n), "y": cy + self.rng.normal(0, s, n),
            "gt_hotspot_id": np.repeat([f"INT-{i:03d}" for i in range(k)], counts),
            "gt_hotspot_type": "intersection"})

    def _corridors(self, n: int) -> pd.DataFrame:
        e = self.edges
        art = e[e["road_class"].isin(ARTERIAL_CLASSES) & e["road_name"].notna()]
        if n == 0 or art.empty:
            return pd.DataFrame(columns=["x", "y", "gt_hotspot_id", "gt_hotspot_type"])
        load = art["road_class"].map(ROAD_CLASS_LOAD).to_numpy(float)
        lines, used_names = [], set()
        for idx in self.rng.choice(len(art), size=len(art), replace=False, p=load / load.sum()):
            if len(lines) == self.cfg.n_planted_corridors:
                break
            name = art["road_name"].iloc[idx]
            if name in used_names:               # one corridor per named road
                continue
            target = self.rng.uniform(*self.cfg.corridor_length_m)
            path = walk_named_road(self.G, art.index[idx], target, self.rng)
            line = shapely.line_merge(shapely.union_all(e.loc[path, "geometry"].to_numpy()))
            if line.geom_type == "LineString" and line.length >= 0.6 * target:
                lines.append(line)
                used_names.add(name)
        if not lines:
            return pd.DataFrame(columns=["x", "y", "gt_hotspot_id", "gt_hotspot_type"])

        counts = self._intensities(len(lines), n)
        geoms = np.repeat(np.array(lines, dtype=object), counts)
        frac = self.rng.random(n)
        pts = shapely.line_interpolate_point(geoms, frac, normalized=True)
        # lateral offset perpendicular-ish (isotropic, small) off the centreline
        s = self.cfg.corridor_spread_m
        return pd.DataFrame({
            "x": shapely.get_x(pts) + self.rng.normal(0, s, n),
            "y": shapely.get_y(pts) + self.rng.normal(0, s, n),
            "gt_hotspot_id": np.repeat([f"COR-{i:03d}" for i in range(len(lines))], counts),
            "gt_hotspot_type": "corridor"})


class CollisionDataGenerator:
    """
    Generate a synthetic, spatially and temporally realistic LA collision table.

    Parameters
    ----------
    gen_cfg, area_cfg, crs_cfg:
        Configuration blocks; default to the global ``settings``.
    seed:
        Overrides ``gen_cfg.random_seed`` for reproducible runs.
    """

    def __init__(
        self,
        gen_cfg: GeneratorConfig = settings.generator,
        area_cfg: StudyAreaConfig = settings.study_area,
        crs_cfg: CRSConfig = settings.crs,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = gen_cfg
        self.crs_cfg = crs_cfg
        self.area_cfg = area_cfg
        self.rng = np.random.default_rng(gen_cfg.random_seed if seed is None else seed)
        self._to_utm = Transformer.from_crs(crs_cfg.geographic, crs_cfg.projected, always_xy=True)
        self._to_geo = Transformer.from_crs(crs_cfg.projected, crs_cfg.geographic, always_xy=True)
        self._land = Polygon(area_cfg.land_polygon)
        shapely.prepare(self._land)
        self._areas = self._project_areas()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self, n: Optional[int] = None,
                 graph: Optional[nx.MultiDiGraph] = None) -> pd.DataFrame:
        """
        Return a DataFrame of ``n`` (default ``cfg.n_records``) collision records.

        If ``graph`` is given, ``intersection_share`` + ``corridor_share`` of the
        records are planted on the road network (see ``NetworkHotspotPlanter``).
        """
        n = n or self.cfg.n_records
        with timed(logger, f"Generating {n:,} synthetic collisions"):
            loc = self._sample_locations(n, graph)
            ts = self._sample_timestamps(n)
            df = loc.assign(timestamp=ts.to_numpy())
            df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
            df.insert(0, "collision_id", self._make_ids(df["timestamp"]))

            df = df.join(self._sample_conditions(df["timestamp"]))
            df = df.join(self._sample_driver(n))
            df = df.join(self._sample_vehicle(df["timestamp"].dt.year.to_numpy()))
            df = df.join(self._sample_behaviour(df))
            df = df.join(self._sample_outcomes(df))
            df = self._inject_dirt(df)
        logger.info("Severity mix: %s",
                    df["severity"].value_counts(normalize=True).round(3).to_dict())
        return df

    @staticmethod
    def save(df: pd.DataFrame, path: Path) -> Path:
        """Write ``df`` to CSV (ISO timestamps) and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
        logger.info("Wrote %s rows to %s", f"{len(df):,}", path)
        return path

    # ------------------------------------------------------------------ #
    # Space
    # ------------------------------------------------------------------ #
    def _project_areas(self) -> list[_ProjectedArea]:
        lo, hi = self.cfg.hotspot_sigma_m
        areas = []
        for name, (lat, lon) in self.area_cfg.lapd_areas.items():
            x, y = self._to_utm.transform(lon, lat)
            areas.append(_ProjectedArea(name, x, y,
                                        self.cfg.area_weights.get(name, 1.0),
                                        float(self.rng.uniform(lo, hi))))
        return areas

    def _sample_locations(self, n: int,
                          graph: Optional[nx.MultiDiGraph] = None) -> pd.DataFrame:
        """Background mixture plus (optionally) network-planted hotspots, with LAPD labels."""
        planted = pd.DataFrame(columns=["x", "y", "gt_hotspot_id", "gt_hotspot_type"])
        if graph is not None:
            n_int = int(round(n * self.cfg.intersection_share))
            n_cor = int(round(n * self.cfg.corridor_share))
            planted = NetworkHotspotPlanter(graph, self.cfg, self.crs_cfg, self.rng) \
                .sample(n_int, n_cor)
            logger.info("Planted %d intersection + %d corridor crashes on the network",
                        (planted["gt_hotspot_type"] == "intersection").sum(),
                        (planted["gt_hotspot_type"] == "corridor").sum())
        bx, by = self._sample_background_xy(n - len(planted))
        xs = np.concatenate([planted["x"].to_numpy(float), bx])
        ys = np.concatenate([planted["y"].to_numpy(float), by])
        gt_id = np.concatenate([planted["gt_hotspot_id"].to_numpy(object),
                                np.full(bx.size, "none", dtype=object)])
        gt_type = np.concatenate([planted["gt_hotspot_type"].to_numpy(object),
                                  np.full(bx.size, "background", dtype=object)])

        area = lapd_area_for(xs, ys, self.area_cfg, self.crs_cfg)
        lon, lat = self._to_geo.transform(xs, ys)
        out = pd.DataFrame({"Latitude": np.round(lat, 6), "Longitude": np.round(lon, 6),
                            "area_name": area, "gt_hotspot_id": gt_id,
                            "gt_hotspot_type": gt_type})
        return out.sample(frac=1.0, random_state=int(self.rng.integers(2**31))) \
                  .reset_index(drop=True)

    def _sample_background_xy(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Area-hotspot mixture + uniform background, rejection-sampled onto land."""
        xs, ys = np.empty(0), np.empty(0)
        w, s, e, nth = self.area_cfg.bbox
        weights = np.array([a.weight for a in self._areas])
        weights = weights / weights.sum()
        cx = np.array([a.x for a in self._areas])
        cy = np.array([a.y for a in self._areas])
        sig = np.array([a.sigma_m for a in self._areas])

        while xs.size < n:
            m = int((n - xs.size) * 1.3) + 100
            hot = self.rng.random(m) < self.cfg.hotspot_share
            idx = self.rng.choice(len(self._areas), size=m, p=weights)
            x = cx[idx] + self.rng.normal(0, 1, m) * sig[idx]
            y = cy[idx] + self.rng.normal(0, 1, m) * sig[idx]
            # Background: uniform in the geographic bbox.
            blon = self.rng.uniform(w, e, m)
            blat = self.rng.uniform(s, nth, m)
            bx, by = self._to_utm.transform(blon, blat)
            x = np.where(hot, x, bx)
            y = np.where(hot, y, by)
            # GPS jitter in metres
            x += self.rng.normal(0, self.cfg.gps_jitter_m, m)
            y += self.rng.normal(0, self.cfg.gps_jitter_m, m)

            lon, lat = self._to_geo.transform(x, y)
            keep = shapely.contains_xy(self._land, lon, lat)
            xs, ys = np.concatenate([xs, x[keep]]), np.concatenate([ys, y[keep]])

        return xs[:n], ys[:n]

    # ------------------------------------------------------------------ #
    # Time
    # ------------------------------------------------------------------ #
    def _sample_timestamps(self, n: int) -> pd.Series:
        days = pd.date_range(self.cfg.start_date, self.cfg.end_date, freq="D")
        p_day = _DOW_WEIGHT[days.dayofweek.to_numpy()]
        day = days[self.rng.choice(len(days), size=n, p=p_day / p_day.sum())]
        weekend = day.dayofweek.to_numpy() >= 5

        hours = np.empty(n, dtype=int)
        for mask, profile in ((~weekend, _HOUR_WEEKDAY), (weekend, _HOUR_WEEKEND)):
            hours[mask] = self.rng.choice(24, size=mask.sum(), p=profile / profile.sum())
        minutes = self.rng.integers(0, 60, n)
        offset = pd.to_timedelta(hours * 60 + minutes, unit="min")
        return pd.Series(day + offset, name="timestamp")

    @staticmethod
    def _make_ids(ts: pd.Series) -> pd.Series:
        """LAPD-style DR numbers: YY + 7-digit sequence within year."""
        yy = (ts.dt.year % 100).astype(str).str.zfill(2)
        seq = ts.groupby(ts.dt.year).cumcount().add(1).astype(str).str.zfill(7)
        return "DR" + yy + seq

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    def _sample_conditions(self, ts: pd.Series) -> pd.DataFrame:
        """Weather, lighting and road surface consistent with date and hour."""
        n = len(ts)
        doy = ts.dt.dayofyear.to_numpy()
        month = ts.dt.month.to_numpy()
        hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60

        # Southern California: rain concentrated Nov–Mar, June-gloom fog.
        rainy = np.isin(month, [11, 12, 1, 2, 3])
        p_rain = np.where(rainy, 0.10, 0.01)
        p_fog = np.where(np.isin(month, [5, 6, 7]) & (hour < 10), 0.08, 0.02)
        u = self.rng.random(n)
        weather = np.select([u < p_rain, u < p_rain + p_fog, u < p_rain + p_fog + 0.12],
                            ["rain", "fog", "cloudy"], "clear")

        lit = self.rng.random(n) < 0.85
        lighting = lighting_from_time(ts, street_lights=lit)

        wet = np.where(weather == "rain", self.rng.random(n) < 0.95,
                       self.rng.random(n) < 0.02)
        surface = np.where(wet, "wet", "dry")
        return pd.DataFrame({"weather": weather, "lighting": lighting, "road_surface": surface})

    # ------------------------------------------------------------------ #
    # People & vehicles
    # ------------------------------------------------------------------ #
    def _sample_driver(self, n: int) -> pd.DataFrame:
        young = self.rng.random(n) < 0.15
        age = np.where(young, self.rng.normal(21, 3, n), self.rng.normal(41, 14, n))
        age = np.clip(np.round(age), 16, 92).astype(int)
        gender = self.rng.choice(["M", "F", "X"], size=n, p=[0.58, 0.40, 0.02])
        licence_years = np.clip(age - 16 - self.rng.geometric(0.35, n) + 1, 0, None)
        prior_claims = self.rng.poisson(0.12 + 0.18 * (age < 25) + 0.05 * (age > 75))
        return pd.DataFrame({"driver_age": age, "driver_gender": gender,
                             "license_years": licence_years,
                             "prior_claims_3yr": prior_claims})

    def _sample_vehicle(self, collision_year: np.ndarray) -> pd.DataFrame:
        n = len(collision_year)
        vtype = self.rng.choice(list(_VEHICLE_TYPES), size=n,
                                p=np.array(list(_VEHICLE_TYPES.values())))
        vehicle_age = np.clip(np.round(self.rng.gamma(2.2, 4.0, n)), 0, 30).astype(int)
        return pd.DataFrame({"vehicle_type": vtype,
                             "vehicle_year": collision_year - vehicle_age})

    def _sample_behaviour(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        hour = df["timestamp"].dt.hour.to_numpy()
        weekend = df["timestamp"].dt.dayofweek.to_numpy() >= 5
        night = (hour >= 22) | (hour <= 3)
        young = df["driver_age"].to_numpy() < 25

        p_speed = 0.08 * np.where(young, 1.8, 1.0) * np.where(night, 1.3, 1.0)
        p_alcohol = 0.025 * np.where(night & weekend, 4.0, np.where(night, 2.0, 1.0))
        p_distract = np.where(young, 0.16, 0.09)

        ctype = self.rng.choice(list(_COLLISION_TYPES), size=n,
                                p=np.array(list(_COLLISION_TYPES.values())))
        if "gt_hotspot_type" in df:
            for kind, mix in (("intersection", _INTERSECTION_TYPES), ("corridor", _CORRIDOR_TYPES)):
                m = (df["gt_hotspot_type"] == kind).to_numpy()
                ctype[m] = self.rng.choice(list(mix), size=m.sum(), p=list(mix.values()))
        single = np.isin(ctype, ["hit_object", "pedestrian"])
        n_veh = np.where(single, 1, 2 + self.rng.poisson(0.25, n))
        return pd.DataFrame({
            "collision_type": ctype,
            "num_vehicles": n_veh,
            "speeding": self.rng.random(n) < p_speed,
            "alcohol_involved": self.rng.random(n) < p_alcohol,
            "distracted": self.rng.random(n) < p_distract,
            "hit_and_run": self.rng.random(n) < 0.15,
        })

    # ------------------------------------------------------------------ #
    # Outcomes
    # ------------------------------------------------------------------ #
    def _sample_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal-logit severity, injury counts and claim amounts."""
        n = len(df)
        age = df["driver_age"].to_numpy()
        veh_age = df["timestamp"].dt.year.to_numpy() - df["vehicle_year"].to_numpy()
        vt, ct, lt = df["vehicle_type"], df["collision_type"], df["lighting"]

        score = (0.9 * df["speeding"] + 1.2 * df["alcohol_involved"]
                 + 0.3 * df["distracted"]
                 + 1.4 * vt.isin(["motorcycle", "bicycle"]) + 0.5 * (vt == "commercial_truck")
                 + 0.9 * (ct == "pedestrian") + 0.7 * (ct == "head_on")
                 + 0.25 * (lt == "dark_street_lights") + 0.55 * (lt == "dark_no_street_lights")
                 + 0.30 * (df["road_surface"] == "wet")
                 + 0.35 * (age < 25) + 0.025 * np.clip(age - 65, 0, None)
                 + 0.015 * veh_age).to_numpy(dtype=float)

        latent = score + self.rng.logistic(0, 1, n)
        cuts = np.quantile(latent, np.cumsum(_SEVERITY_SHARES)[:-1])
        sev_idx = np.searchsorted(cuts, latent)
        severity = np.array(_SEVERITY_LEVELS)[sev_idx]

        injured = np.select([sev_idx == 1, sev_idx == 2, sev_idx == 3],
                            [1 + self.rng.poisson(0.3, n), 1 + self.rng.poisson(0.5, n),
                             self.rng.poisson(0.8, n)], 0)
        killed = np.where(sev_idx == 3, 1 + self.rng.poisson(0.1, n), 0)

        value_factor = np.clip(1.0 + 0.03 * (12 - veh_age), 0.7, 1.4)
        commercial = np.where(vt.isin(["commercial_truck", "bus"]), 1.35, 1.0)
        claim = (np.exp(np.log(_CLAIM_MEDIAN_USD[sev_idx])
                        + self.rng.normal(0, 1, n) * _CLAIM_SIGMA[sev_idx])
                 * value_factor * commercial)
        return pd.DataFrame({"severity": severity, "severity_code": sev_idx,
                             "num_injured": injured, "num_killed": killed,
                             "claim_amount_usd": np.round(claim, 2)})

    # ------------------------------------------------------------------ #
    # Data-quality defects
    # ------------------------------------------------------------------ #
    def _inject_dirt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Overwrite a small share of rows with typical feed defects (row count unchanged)."""
        k = int(round(len(df) * self.cfg.dirty_fraction))
        if k == 0:
            return df
        df = df.copy()
        rows = self.rng.choice(len(df), size=k, replace=False)
        null_rows, zero_rows, dup_rows = np.array_split(rows, [int(k * 0.4), int(k * 0.8)])
        df.loc[null_rows, ["Latitude", "Longitude"]] = np.nan
        df.loc[zero_rows, ["Latitude", "Longitude"]] = 0.0
        if dup_rows.size:
            src = self.rng.choice(np.setdiff1d(np.arange(len(df)), rows), size=dup_rows.size)
            df = (pd.concat([df.drop(index=dup_rows), df.loc[src]])
                  .sort_values("timestamp", kind="stable").reset_index(drop=True))
        logger.info("Injected %d defects (%d null, %d zero-coord, %d duplicate)",
                    k, len(null_rows), len(zero_rows), len(dup_rows))
        return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate synthetic LA collision data.")
    parser.add_argument("-n", "--n-records", type=int, default=settings.generator.n_records)
    parser.add_argument("--seed", type=int, default=settings.generator.random_seed)
    parser.add_argument("-o", "--output", type=Path, default=settings.paths.raw_collisions)
    parser.add_argument("--graphml", type=Path, default=None,
                        help="Road graph to plant intersection/corridor hotspots on.")
    args = parser.parse_args(argv)

    settings.paths.ensure()
    graph = ox.load_graphml(args.graphml) if args.graphml else None
    df = CollisionDataGenerator(seed=args.seed).generate(args.n_records, graph=graph)
    return CollisionDataGenerator.save(df, args.output)


if __name__ == "__main__":
    main()
