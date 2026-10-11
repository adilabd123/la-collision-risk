"""
Central configuration for the LA Vehicle Collision Insurance Risk Assessment System.

Every module imports its settings from here, so paths, CRS definitions and
hyperparameters live in exactly one place. Values can be overridden with
environment variables (prefix ``CRS_RISK_``) without editing code, which keeps
the same build usable on a laptop, in CI and in a container.

Example::

    from config import settings
    settings.paths.raw_collisions          # Path to the raw CSV
    settings.crs.projected                 # "EPSG:32611"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_ENV_PREFIX = "CRS_RISK_"


def _env(name: str, default: str) -> str:
    """Return ``CRS_RISK_<name>`` from the environment, or ``default``."""
    return os.getenv(f"{_ENV_PREFIX}{name}", default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


PROJECT_ROOT: Path = Path(_env("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathConfig:
    """Filesystem layout. Directories are created on first access via ``ensure()``."""

    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"

    @property
    def raw_collisions(self) -> Path:
        """Synthetic (or real LAPD) collision records, unprocessed."""
        return self.raw_dir / "la_collisions_raw.csv"

    @property
    def road_graph(self) -> Path:
        """Cached OSM drive network (GraphML, EPSG:4326 as downloaded)."""
        return self.cache_dir / "la_drive_network.graphml"

    @property
    def snapped_collisions(self) -> Path:
        """Ingestion output: cleaned, projected, road-snapped points (GeoParquet)."""
        return self.processed_dir / "la_collisions_snapped.parquet"

    @property
    def features(self) -> Path:
        """Feature-engineering output (GeoParquet)."""
        return self.processed_dir / "la_collisions_features.parquet"

    @property
    def clustered(self) -> Path:
        """Collisions with DBSCAN cluster labels (GeoParquet)."""
        return self.processed_dir / "la_collisions_clustered.parquet"

    @property
    def cluster_summary(self) -> Path:
        """One row per hotspot cluster with footprint polygon (GeoParquet)."""
        return self.processed_dir / "la_hotspot_clusters.parquet"

    @property
    def venues_cache(self) -> Path:
        """Alcohol-serving venue points (simulated or OSM), EPSG:32611."""
        return self.cache_dir / "alcohol_venues.parquet"

    @property
    def model_bundle(self) -> Path:
        """Trained model + preprocessing + hotspot matcher, loaded by the API."""
        return self.models_dir / "risk_model.joblib"

    @property
    def segment_risk(self) -> Path:
        """4-tier spatial risk per road segment (GeoParquet)."""
        return self.processed_dir / "la_segment_risk_tiers.parquet"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def rejected_records(self) -> Path:
        """Rows dropped by validation, kept for audit."""
        return self.interim_dir / "la_collisions_rejected.csv"

    def ensure(self) -> "PathConfig":
        """Create all directories (idempotent) and return self."""
        for p in (self.raw_dir, self.interim_dir, self.processed_dir,
                  self.cache_dir, self.models_dir, self.logs_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- #
# Coordinate reference systems & study area
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CRSConfig:
    """
    Geographic (input) and projected (analysis) CRS.

    EPSG:4326 stores degrees, which are not equal-length units, so distances and
    buffers computed in it are wrong. EPSG:32611 (WGS 84 / UTM Zone 11N) covers
    Los Angeles (central meridian -117°) and uses metres, so every distance,
    snap tolerance and kernel bandwidth in the pipeline is computed there.
    """

    geographic: str = "EPSG:4326"
    projected: str = _env("PROJECTED_CRS", "EPSG:32611")


@dataclass(frozen=True)
class StudyAreaConfig:
    """Bounding box of the City of Los Angeles, plus a coarse land mask."""

    # (west, south, east, north) in EPSG:4326 — OSMnx 2.x bbox order.
    bbox: Tuple[float, float, float, float] = (-118.668, 33.704, -118.155, 34.337)

    # Coarse land polygon (lon, lat) that trims Santa Monica Bay / San Pedro Bay
    # from the bbox so synthetic points do not fall in the ocean.
    land_polygon: Tuple[Tuple[float, float], ...] = (
        (-118.668, 34.337), (-118.155, 34.337), (-118.155, 33.704),
        (-118.280, 33.704), (-118.330, 33.730), (-118.420, 33.770),
        (-118.410, 33.850), (-118.430, 33.930), (-118.470, 33.980),
        (-118.520, 34.030), (-118.600, 34.040), (-118.668, 34.040),
    )

    # Approximate centroids of the 21 LAPD geographic areas (lat, lon).
    # Used to label records with an ``area_name`` compatible with LAPD open data.
    lapd_areas: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Central": (34.044, -118.247), "Rampart": (34.068, -118.276),
        "Southwest": (34.010, -118.305), "Hollenbeck": (34.045, -118.205),
        "Harbor": (33.760, -118.290), "Hollywood": (34.098, -118.330),
        "Wilshire": (34.060, -118.350), "West LA": (34.045, -118.440),
        "Van Nuys": (34.180, -118.450), "West Valley": (34.190, -118.540),
        "Northeast": (34.110, -118.220), "77th Street": (33.970, -118.290),
        "Newton": (34.010, -118.260), "Pacific": (33.980, -118.420),
        "N Hollywood": (34.170, -118.380), "Foothill": (34.260, -118.400),
        "Devonshire": (34.260, -118.530), "Southeast": (33.940, -118.260),
        "Mission": (34.270, -118.460), "Olympic": (34.055, -118.300),
        "Topanga": (34.200, -118.600),
    })


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeneratorConfig:
    """Controls the synthetic 25k-record collision dataset."""

    n_records: int = _env_int("N_RECORDS", 25_000)
    random_seed: int = _env_int("SEED", 42)
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"

    # Share of points drawn from area hotspots vs. a diffuse background.
    hotspot_share: float = 0.85
    hotspot_sigma_m: Tuple[float, float] = (800.0, 2_500.0)   # min/max spread
    gps_jitter_m: float = 12.0                                 # GPS noise

    # Inject real-world dirt (null coords, 0/0 points, duplicates) so the
    # ingestion validators are exercised. LAPD open data contains all three.
    dirty_fraction: float = _env_float("DIRTY_FRACTION", 0.005)

    # Network-aware planting (used when a road graph is passed to the generator).
    # Real crashes concentrate at complex intersections and along arterials;
    # these shares are carved out of the total and placed on the graph, and
    # carry a ground-truth ``gt_hotspot_id`` for validating the clustering.
    intersection_share: float = 0.20
    corridor_share: float = 0.18
    n_planted_intersections: int = 80
    n_planted_corridors: int = 30
    corridor_length_m: Tuple[float, float] = (900.0, 2_800.0)
    intersection_spread_m: float = 14.0        # approach-lane scatter
    corridor_spread_m: float = 6.0             # lateral scatter off centreline

    # Relative crash intensity per LAPD area (unlisted areas default to 1.0).
    area_weights: Dict[str, float] = field(default_factory=lambda: {
        "Central": 2.2, "77th Street": 1.8, "Southwest": 1.6, "Hollywood": 1.6,
        "Wilshire": 1.5, "Olympic": 1.5, "Newton": 1.4, "Van Nuys": 1.3,
        "Southeast": 1.3, "Rampart": 1.3, "N Hollywood": 1.2,
    })


# --------------------------------------------------------------------------- #
# Ingestion & road snapping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class APIConfig:
    """Serving settings (all overridable via CRS_RISK_* env vars)."""

    model_path: str = _env("MODEL_PATH", "")          # default: paths.model_bundle
    graphml_path: str = _env("GRAPHML", "")           # default: paths.road_graph
    max_batch: int = _env_int("MAX_BATCH", 500)
    timezone: str = _env("TZ_NAME", "America/Los_Angeles")


@dataclass(frozen=True)
class IngestionConfig:
    """Parameters for cleaning, reprojection and OSMnx network snapping."""

    network_type: str = "drive"
    # Points further than this from any road edge are kept but flagged as
    # low-confidence snaps (likely parking lots, private roads or bad geocodes).
    max_snap_distance_m: float = _env_float("MAX_SNAP_M", 50.0)
    # Hard ceiling: beyond this the point is considered a geocoding failure
    # and moved to the rejected-records file.
    reject_snap_distance_m: float = _env_float("REJECT_SNAP_M", 250.0)
    # OSM edge attributes carried onto each collision.
    edge_attributes: Tuple[str, ...] = (
        "osmid", "name", "highway", "maxspeed", "lanes", "oneway", "length",
    )
    drop_duplicates_on: Tuple[str, ...] = ("collision_id",)


# --------------------------------------------------------------------------- #
# Feature engineering & enrichment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureConfig:
    """Temporal encoding, simulated API enrichment and collinearity screening."""

    enrichment_seed: int = _env_int("ENRICH_SEED", 2026)
    rush_hours: Tuple[int, ...] = (7, 8, 9, 16, 17, 18)
    night_hours: Tuple[int, ...] = (21, 22, 23, 0, 1, 2, 3, 4)
    diurnal_harmonics: int = 2                    # 24 h + 12 h cycles

    # POI density
    poi_buffer_m: float = 200.0
    n_simulated_venues: int = 4_000
    venue_source: str = _env("VENUE_SOURCE", "simulated")   # "simulated" | "osm"
    osm_alcohol_tags: Dict[str, List[str]] = field(default_factory=lambda: {
        "amenity": ["bar", "pub", "nightclub", "biergarten"],
        "shop": ["alcohol", "wine", "beer"],
    })

    # Intersection complexity
    intersection_min_legs: int = 3
    at_intersection_m: float = 25.0
    intersection_decay_m: float = 40.0

    # Collinearity screening
    corr_threshold: float = 0.85
    vif_threshold: float = 10.0


# --------------------------------------------------------------------------- #
# DBSCAN hotspot clustering
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClusteringConfig:
    """
    Two-scale DBSCAN search space and hotspot-selection rules.

    A single (eps, min_samples) cannot capture both crash geometries: a
    dangerous junction is a dense knot (tens of crashes within ~20 m), while a
    dangerous corridor is a sparser line (a few crashes per 100 m over 1–3 km).
    Stage 1 tunes a tight DBSCAN for knots; stage 2 re-runs DBSCAN with a
    wider radius on the stage-1 noise and rewards clusters that follow one road.

    Both stages maximise ``capture_score = coverage × ln(PAI)``, where PAI
    (Prediction Accuracy Index) = share of crashes captured / share of city area
    covered. Pure PAI rewards ever-smaller knots and fragments corridors; pure
    coverage rewards chaining everything. The product, with PAI on a log scale,
    balances the two, subject to minimum-PAI, coverage, cluster-count and
    "no mega-cluster" constraints.
    """

    # Stage 1 — intersections / compact knots
    intersection_eps_grid_m: Tuple[float, ...] = (15, 20, 25, 35, 50)   # junction scale
    intersection_min_samples_grid: Tuple[int, ...] = (6, 8, 10, 15, 20, 25, 30)
    intersection_min_coverage: float = 0.08
    intersection_min_clusters: int = 10

    # Stage 2 — corridors, run on stage-1 noise
    corridor_eps_grid_m: Tuple[float, ...] = (40, 60, 80, 100, 130, 160, 200, 250)
    corridor_min_samples_grid: Tuple[int, ...] = (4, 5, 6, 8, 12, 16)
    corridor_min_coverage: float = 0.01           # of ALL crashes, in corridor-like clusters
    corridor_min_clusters: int = 5

    min_pai: float = 50.0                         # hotspots must be ≥ 50× denser than city
    # Smaller DBSCAN clusters are returned to noise: over a multi-year window a
    # handful of crashes on one block is not a defensible hotspot.
    min_hotspot_crashes: int = 10
    knn_k: int = 16                               # k for the k-distance curve (reporting)
    footprint_cell_m: float = 25.0                # raster cell for hotspot area
    max_largest_cluster_share: float = 0.05       # no cluster > 5 % of crashes

    # Cluster typing (used in stage-2 scoring and in final profiling)
    corridor_min_linearity: float = 0.90          # 1 − λ2/λ1 from PCA (straight roads)
    corridor_max_width_m: float = 60.0            # √(12·λ2): crashes hug the centreline
    corridor_min_length_eps: float = 4.0          # length ≥ 4·eps
    corridor_min_road_share: float = 0.60         # dominant named road share …
    corridor_min_segments: int = 3                # … across ≥ 3 road segments (curvy roads)
    # Local contrast: corridor density must be ≥ N× the crash density in a ring
    # around it. Large-eps DBSCAN can chain random background into long thin
    # runs that pass the shape tests; real corridors stand out from their
    # surroundings, chance chains do not.
    corridor_min_local_lift: float = 3.0
    local_ring_m: float = 500.0
    intersection_max_dist_m: float = 40.0         # median dist to junction
    intersection_max_length_m: float = 120.0


# --------------------------------------------------------------------------- #
# Modelling hyperparameters (consumed in Step 3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    """Severity-risk models, imbalance handling, thresholds and risk tiers."""

    # Target: High_Risk = 1 for severe injury or fatal (severity_code >= 2)
    high_risk_min_severity: int = 2

    # Which inputs the model may use.
    #   underwriting   – known before any crash: place, time, conditions,
    #                    driver, vehicle. This is what a quote/pricing API sees.
    #   claims_triage  – adds facts only known after a crash is reported
    #                    (collision type, speeding, alcohol, ...), for
    #                    prioritising incoming claims.
    feature_set: str = _env("FEATURE_SET", "underwriting")
    post_crash_columns: List[str] = field(default_factory=lambda: [
        "collision_type", "num_vehicles", "speeding", "alcohol_involved",
        "distracted", "hit_and_run",
    ])
    categorical_columns: List[str] = field(default_factory=lambda: [
        "vehicle_type", "lighting", "weather", "road_surface", "road_class",
        "hotspot_type", "collision_type",
    ])
    # Data-quality / bookkeeping fields that must not become predictors.
    excluded_numeric: List[str] = field(default_factory=lambda: [
        "snap_distance_m", "x_utm", "y_utm", "Latitude", "Longitude",
        "edge_u", "edge_v", "edge_key", "osm_way_id", "edge_position",
        "cluster_id", "hotspot_loss_rank", "vehicle_year", "Collision_Hour_Frac",
    ])

    # Validation — spatial blocks keep neighbouring crashes on one side of a split
    test_size: float = 0.2
    spatial_cv_folds: int = 5
    spatial_block_size_m: float = 2_000.0
    random_seed: int = _env_int("SEED", 42)

    # Imbalance: SMOTE on training folds only, up to minority:majority = 0.5;
    # 'balanced' class weights then correct the remaining 1:2 gap. (Oversampling
    # all the way to 1:1 would make 'balanced' weights a no-op.)
    # oversampler: "smote" (on one-hot matrix) | "smotenc" | "none"
    oversampler: str = _env("OVERSAMPLER", "smote")
    smote_sampling_strategy: float = 0.5
    smote_k_neighbors: int = 5

    # Decision threshold: chosen on out-of-fold predictions by maximising
    # F-beta (beta = 2 weights recall 4× precision) while flagging at most
    # max_flag_share of crashes (review capacity); the fixed 0.20 is also
    # reported for comparison.
    threshold_beta: float = 2.0
    max_flag_share: float = 0.35
    fixed_threshold: float = 0.20
    threshold_grid: Tuple[float, ...] = tuple(round(0.05 + 0.01 * i, 2) for i in range(76))

    # 4-tier risk model: cut points are quantiles of out-of-fold probabilities.
    tier_labels: Tuple[str, ...] = ("Low", "Moderate", "High", "Very High")
    tier_quantiles: Tuple[float, ...] = (0.50, 0.80, 0.95)
    segment_min_crashes: int = 5        # segments with fewer get shrunk to the mean
    segment_shrinkage_k: float = 10.0   # empirical-Bayes prior strength (crashes)

    lr_params: Dict[str, float] = field(default_factory=lambda: {
        "C": 0.5, "max_iter": 2000})
    rf_params: Dict[str, float] = field(default_factory=lambda: {
        "n_estimators": 400, "max_depth": 14, "min_samples_leaf": 10,
        "max_features": "sqrt", "n_jobs": -1})
    xgb_params: Dict[str, float] = field(default_factory=lambda: {
        "n_estimators": 500, "learning_rate": 0.04, "max_depth": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "reg_lambda": 1.0, "tree_method": "hist", "n_jobs": -1, "eval_metric": "aucpr"})

    # Columns that must never be used as rating features. California
    # (Prop 103 regulations) prohibits gender as an auto-rating factor; they
    # remain in the data only for fairness auditing.
    protected_attributes: List[str] = field(
        default_factory=lambda: ["driver_gender"]
    )
    # Never usable as model inputs: identifiers, targets, outcome fields and
    # synthetic ground truth.
    non_feature_columns: List[str] = field(default_factory=lambda: [
        "collision_id", "severity", "severity_code", "num_injured", "num_killed",
        "claim_amount_usd", "gt_hotspot_id", "gt_hotspot_type", "High_Risk",
    ])


# --------------------------------------------------------------------------- #
# Aggregate settings object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Single entry point for all configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    crs: CRSConfig = field(default_factory=CRSConfig)
    study_area: StudyAreaConfig = field(default_factory=StudyAreaConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log_level: str = _env("LOG_LEVEL", "INFO")


settings = Settings()
