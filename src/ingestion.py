"""
Production ingestion: validate → reproject → snap to the OSM road network.

Pipeline
--------
1. **Load & validate** (``CollisionLoader``) – schema check, type coercion,
   rejection of null / (0, 0) / out-of-area coordinates and duplicate IDs.
   Rejected rows are written to an audit file with a reason code, never
   silently dropped.
2. **Reproject** (``SpatialProjector``) – EPSG:4326 degrees → EPSG:32611
   (UTM 11N) metres, so every downstream distance is in metres.
3. **Road network** (``RoadNetworkLoader``) – OSMnx drive graph for the
   study area, cached to GraphML after the first download and projected to
   the same UTM CRS.
4. **Snap** (``RoadSnapper``) – each crash is matched to its nearest road
   *edge* (not node) with OSMnx, then moved to the closest point *on* that
   edge's true geometry. Road attributes (class, speed limit, lanes, one-way)
   are joined on, and each snap gets a distance and a quality flag.
5. **Persist** – GeoParquet with the snapped point as the active geometry and
   the original GPS point kept as ``raw_geometry``.

Usage::

    python -m src.ingestion                          # download/cache LA network
    python -m src.ingestion --graphml my_graph.graphml
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely

from config import settings
from config.config import CRSConfig, IngestionConfig, PathConfig, StudyAreaConfig
from src.utils import get_logger, timed

logger = get_logger(__name__)

# UTM northings/eastings valid for zone 11N over southern California.
_UTM_EASTING_RANGE = (166_000.0, 834_000.0)
_UTM_NORTHING_RANGE = (3_500_000.0, 4_000_000.0)


# --------------------------------------------------------------------------- #
# 1. Load & validate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColumnSpec:
    """Expected column and the dtype it should be coerced to."""

    name: str
    dtype: str          # "float", "int", "str", "bool", "datetime"
    required: bool = True


DEFAULT_SCHEMA: Tuple[ColumnSpec, ...] = (
    ColumnSpec("collision_id", "str"),
    ColumnSpec("Latitude", "float"),
    ColumnSpec("Longitude", "float"),
    ColumnSpec("timestamp", "datetime"),
    ColumnSpec("area_name", "str", required=False),
    ColumnSpec("driver_age", "int", required=False),
    ColumnSpec("driver_gender", "str", required=False),
    ColumnSpec("vehicle_type", "str", required=False),
    ColumnSpec("severity", "str", required=False),
    ColumnSpec("claim_amount_usd", "float", required=False),
    ColumnSpec("speeding", "bool", required=False),
    ColumnSpec("alcohol_involved", "bool", required=False),
)


class SchemaError(ValueError):
    """Raised when required columns are missing from the input."""


class CollisionLoader:
    """Read raw collision records and split them into valid and rejected sets."""

    def __init__(
        self,
        schema: Sequence[ColumnSpec] = DEFAULT_SCHEMA,
        area_cfg: StudyAreaConfig = settings.study_area,
        ingest_cfg: IngestionConfig = settings.ingestion,
    ) -> None:
        self.schema = tuple(schema)
        self.bbox = area_cfg.bbox
        self.dedupe_cols = list(ingest_cfg.drop_duplicates_on)

    def load(self, path: Path) -> pd.DataFrame:
        """Read CSV or Parquet from ``path``."""
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        reader = pd.read_parquet if path.suffix == ".parquet" else pd.read_csv
        df = reader(path)
        logger.info("Loaded %s rows from %s", f"{len(df):,}", path.name)
        return df

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Coerce types and apply row-level rules.

        Returns
        -------
        (valid, rejected)
            ``rejected`` carries a ``reject_reason`` column.
        """
        missing = [c.name for c in self.schema if c.required and c.name not in df.columns]
        if missing:
            raise SchemaError(f"Missing required columns: {missing}")

        df = self._coerce(df.copy())
        reasons = pd.Series(pd.NA, index=df.index, dtype="string")

        def flag(mask: pd.Series, reason: str) -> None:
            reasons.loc[mask & reasons.isna()] = reason

        lat, lon = df["Latitude"], df["Longitude"]
        w, s, e, n = self.bbox
        flag(lat.isna() | lon.isna(), "null_coordinates")
        flag((lat == 0) & (lon == 0), "zero_coordinates")
        flag(~lat.between(s, n) | ~lon.between(w, e), "outside_study_area")
        flag(df["timestamp"].isna(), "invalid_timestamp")
        flag(df["collision_id"].isna(), "missing_id")
        flag(df.duplicated(subset=self.dedupe_cols, keep="first"), "duplicate_id")

        rejected = df[reasons.notna()].assign(reject_reason=reasons[reasons.notna()])
        valid = df[reasons.isna()].reset_index(drop=True)
        if len(rejected):
            logger.warning("Rejected %d rows: %s", len(rejected),
                           rejected["reject_reason"].value_counts().to_dict())
        return valid, rejected

    def _coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        for spec in self.schema:
            if spec.name not in df.columns:
                continue
            col = df[spec.name]
            if spec.dtype == "float":
                df[spec.name] = pd.to_numeric(col, errors="coerce")
            elif spec.dtype == "int":
                df[spec.name] = pd.to_numeric(col, errors="coerce").astype("Int64")
            elif spec.dtype == "datetime":
                df[spec.name] = pd.to_datetime(col, errors="coerce")
            elif spec.dtype == "bool":
                df[spec.name] = col.map(_to_bool).astype("boolean")
            else:
                df[spec.name] = col.astype("string").str.strip()
        return df


def _to_bool(v: object) -> Optional[bool]:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return {"true": True, "1": True, "y": True, "false": False, "0": False, "n": False}.get(
            v.strip().lower())
    if isinstance(v, (int, float)) and not pd.isna(v):
        return bool(v)
    return None


# --------------------------------------------------------------------------- #
# 2. Reproject
# --------------------------------------------------------------------------- #
class SpatialProjector:
    """Convert a lat/lon table into a GeoDataFrame in projected metres."""

    def __init__(self, crs_cfg: CRSConfig = settings.crs) -> None:
        self.src_crs = crs_cfg.geographic
        self.dst_crs = crs_cfg.projected

    def to_projected(self, df: pd.DataFrame,
                     lat_col: str = "Latitude", lon_col: str = "Longitude") -> gpd.GeoDataFrame:
        """Build points in EPSG:4326 and reproject to the analysis CRS."""
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=self.src_crs
        ).to_crs(self.dst_crs)
        gdf["x_utm"] = gdf.geometry.x
        gdf["y_utm"] = gdf.geometry.y
        self._sanity_check(gdf)
        return gdf

    def _sanity_check(self, gdf: gpd.GeoDataFrame) -> None:
        """Fail fast on swapped lat/lon or a wrong zone (coordinates out of range)."""
        if gdf.empty or self.dst_crs != "EPSG:32611":
            return
        xmin, ymin, xmax, ymax = gdf.total_bounds
        if not (_UTM_EASTING_RANGE[0] <= xmin and xmax <= _UTM_EASTING_RANGE[1]
                and _UTM_NORTHING_RANGE[0] <= ymin and ymax <= _UTM_NORTHING_RANGE[1]):
            raise ValueError(
                f"Projected bounds {gdf.total_bounds.round(0)} are outside UTM 11N "
                "southern-California range — check for swapped Latitude/Longitude."
            )


# --------------------------------------------------------------------------- #
# 3. Road network
# --------------------------------------------------------------------------- #
class RoadNetworkLoader:
    """
    Provide a projected OSMnx road graph, downloading once and caching to GraphML.

    The cached file is stored unprojected (as OSM serves it) so it can be
    reprojected to any CRS later without re-downloading.
    """

    def __init__(
        self,
        paths: PathConfig = settings.paths,
        area_cfg: StudyAreaConfig = settings.study_area,
        crs_cfg: CRSConfig = settings.crs,
        ingest_cfg: IngestionConfig = settings.ingestion,
    ) -> None:
        self.cache_path = paths.road_graph
        self.bbox = area_cfg.bbox
        self.dst_crs = crs_cfg.projected
        self.network_type = ingest_cfg.network_type
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(paths.cache_dir / "osmnx_http")
        ox.settings.log_console = False

    def load(self, graphml: Optional[Path] = None, force_download: bool = False) -> nx.MultiDiGraph:
        """
        Return the road graph projected to ``dst_crs``.

        Order of preference: explicit ``graphml`` → cached GraphML → OSM download.
        """
        path = graphml or self.cache_path
        if path.exists() and not force_download:
            with timed(logger, f"Loading road graph from {path.name}"):
                G = ox.load_graphml(path)
        else:
            G = self._download()
        return self.project(G)

    def project(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Project a graph to the analysis CRS if it isn't already."""
        if str(G.graph.get("crs", "")).upper() == self.dst_crs.upper():
            return G
        with timed(logger, f"Projecting graph to {self.dst_crs}"):
            return ox.project_graph(G, to_crs=self.dst_crs)

    def _download(self) -> nx.MultiDiGraph:
        logger.info("Downloading OSM '%s' network for bbox %s (first run can take minutes)",
                    self.network_type, self.bbox)
        with timed(logger, "OSM download"):
            G = ox.graph_from_bbox(self.bbox, network_type=self.network_type,
                                   simplify=True, truncate_by_edge=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, self.cache_path)
        logger.info("Cached graph (%d nodes, %d edges) to %s",
                    G.number_of_nodes(), G.number_of_edges(), self.cache_path)
        return G


# --------------------------------------------------------------------------- #
# 4. Snap
# --------------------------------------------------------------------------- #
class RoadSnapper:
    """
    Snap projected points onto the nearest road edge of an OSMnx graph.

    The edge GeoDataFrame (``ox.graph_to_gdfs``, true curved geometries) and
    an STRtree over it are built once; each call runs the same nearest-edge
    query as ``ox.distance.nearest_edges``, which would otherwise rebuild
    both from the graph on every call (seconds per request on a city graph).
    Shapely's linear referencing then places the point at the closest
    location on that edge's geometry (not just its end nodes).
    """

    def __init__(self, G: nx.MultiDiGraph,
                 ingest_cfg: IngestionConfig = settings.ingestion) -> None:
        if not ox.projection.is_projected(G.graph.get("crs")):
            raise ValueError("RoadSnapper needs a projected graph (metres); "
                             "use RoadNetworkLoader.project() first.")
        self.G = G
        self.cfg = ingest_cfg
        self.edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        self._lines = self.edges.geometry.to_numpy()
        self._keys = self.edges.index.to_numpy()          # (u, v, key) tuples
        self._tree = shapely.STRtree(self._lines)
        logger.info("Snapper ready: %s edges", f"{len(self.edges):,}")

    def snap(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Return ``gdf`` with the active geometry moved onto the road network.

        Added columns: ``edge_u``, ``edge_v``, ``edge_key``, ``snap_distance_m``,
        ``edge_position`` (0–1 along the edge), ``snap_quality``
        (``good`` / ``low_confidence``), ``raw_geometry`` and normalised road
        attributes (``road_class``, ``speed_limit_mph``, ``lanes``, ...).
        """
        if gdf.crs is None or not gdf.crs.equals(self.edges.crs):
            raise ValueError(f"Point CRS {gdf.crs} does not match graph CRS {self.edges.crs}")

        with timed(logger, f"Snapping {len(gdf):,} points"):
            pts = gdf.geometry.to_numpy()
            (_, pos), dist = self._tree.query_nearest(pts, return_distance=True,
                                                      all_matches=False)
            uvk = np.array([tuple(t) for t in self._keys[pos]], dtype=np.int64)
            lines = self._lines[pos]

            along = shapely.line_locate_point(lines, pts)
            snapped = shapely.line_interpolate_point(lines, along)
            lengths = shapely.length(lines)

        out = gdf.copy()
        out["raw_geometry"] = gpd.GeoSeries(pts, crs=gdf.crs, index=gdf.index)
        out["edge_u"], out["edge_v"], out["edge_key"] = uvk[:, 0], uvk[:, 1], uvk[:, 2]
        out["snap_distance_m"] = np.round(dist, 3)
        out["edge_position"] = np.where(lengths > 0, along / np.maximum(lengths, 1e-9), 0.0)
        out["snap_quality"] = np.where(dist <= self.cfg.max_snap_distance_m,
                                       "good", "low_confidence")
        out = out.join(self._edge_attributes(uvk, index=gdf.index))
        out[out.geometry.name] = gpd.GeoSeries(snapped, crs=gdf.crs, index=gdf.index)
        return out

    # ---------------------------- attributes ------------------------------ #
    def _edge_attributes(self, uvk: np.ndarray, index: pd.Index) -> pd.DataFrame:
        cols = [c for c in self.cfg.edge_attributes if c in self.edges.columns]
        raw = self.edges.loc[list(map(tuple, uvk)), cols].reset_index(drop=True)
        attrs = pd.DataFrame(index=raw.index)
        if "osmid" in raw:
            attrs["osm_way_id"] = raw["osmid"].map(lambda v: _first(v)).astype("Int64")
        if "name" in raw:
            attrs["road_name"] = raw["name"].map(lambda v: "; ".join(map(str, _as_list(v)))
                                                 if _present(v) else pd.NA)
        if "highway" in raw:
            attrs["road_class"] = raw["highway"].map(_first).astype("string")
        if "maxspeed" in raw:
            attrs["speed_limit_mph"] = raw["maxspeed"].map(parse_maxspeed_mph)
        if "lanes" in raw:
            attrs["lanes"] = raw["lanes"].map(_max_numeric).astype("Int64")
        if "oneway" in raw:
            attrs["oneway"] = raw["oneway"].map(_to_bool).astype("boolean")
        if "length" in raw:
            attrs["edge_length_m"] = pd.to_numeric(raw["length"], errors="coerce").round(2)
        attrs.index = index
        return attrs


# ---- OSM tag normalisation (tags may be scalars, lists, or strings) ------- #
def _present(v: object) -> bool:
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def _as_list(v: object) -> List[object]:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _first(v: object) -> object:
    return _as_list(v)[0] if _present(v) else pd.NA


def _max_numeric(v: object) -> Optional[float]:
    if not _present(v):
        return None
    nums = [float(m) for x in _as_list(v) for m in re.findall(r"\d+(?:\.\d+)?", str(x))]
    return max(nums) if nums else None


def parse_maxspeed_mph(v: object) -> Optional[float]:
    """Parse OSM ``maxspeed`` ("35 mph", "50", ["25 mph", "35 mph"]) to mph (max value)."""
    if not _present(v):
        return None
    speeds = []
    for item in _as_list(v):
        s = str(item).lower()
        m = re.search(r"\d+(?:\.\d+)?", s)
        if not m:
            continue
        val = float(m.group())
        # US OSM data is mostly tagged "xx mph"; a bare number means km/h per OSM spec.
        speeds.append(val if "mph" in s else round(val * 0.621371, 1))
    return max(speeds) if speeds else None


# --------------------------------------------------------------------------- #
# 5. Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class IngestionResult:
    """Outputs and run statistics of an ingestion run."""

    data: gpd.GeoDataFrame
    rejected: pd.DataFrame
    stats: Dict[str, float] = field(default_factory=dict)


class IngestionPipeline:
    """Run load → validate → reproject → snap → persist end to end."""

    def __init__(
        self,
        paths: PathConfig = settings.paths,
        ingest_cfg: IngestionConfig = settings.ingestion,
        loader: Optional[CollisionLoader] = None,
        projector: Optional[SpatialProjector] = None,
        network_loader: Optional[RoadNetworkLoader] = None,
    ) -> None:
        self.paths = paths
        self.cfg = ingest_cfg
        self.loader = loader or CollisionLoader()
        self.projector = projector or SpatialProjector()
        self.network_loader = network_loader or RoadNetworkLoader()

    def run(
        self,
        input_path: Optional[Path] = None,
        graph: Optional[nx.MultiDiGraph] = None,
        graphml: Optional[Path] = None,
        write: bool = True,
    ) -> IngestionResult:
        """
        Execute the pipeline.

        Parameters
        ----------
        input_path: raw CSV/Parquet; defaults to ``paths.raw_collisions``.
        graph: an in-memory graph (e.g. for tests); skips loading/downloading.
        graphml: explicit GraphML file to use instead of the cache.
        write: persist outputs to ``paths.processed_dir`` / ``interim_dir``.
        """
        raw = self.loader.load(input_path or self.paths.raw_collisions)
        valid, rejected = self.loader.validate(raw)
        points = self.projector.to_projected(valid)

        G = self.network_loader.project(graph) if graph is not None \
            else self.network_loader.load(graphml)
        snapped = RoadSnapper(G, self.cfg).snap(points)

        too_far = snapped["snap_distance_m"] > self.cfg.reject_snap_distance_m
        if too_far.any():
            far = pd.DataFrame(snapped.loc[too_far].drop(columns=["geometry", "raw_geometry"]))
            rejected = pd.concat([rejected, far.assign(reject_reason="no_road_within_tolerance")],
                                 ignore_index=True)
            snapped = snapped.loc[~too_far].reset_index(drop=True)

        stats = self._stats(raw, snapped, rejected)
        if write:
            self._write(snapped, rejected)
        return IngestionResult(snapped, rejected, stats)

    @staticmethod
    def _stats(raw: pd.DataFrame, snapped: gpd.GeoDataFrame, rejected: pd.DataFrame) -> Dict[str, float]:
        d = snapped["snap_distance_m"]
        stats = {
            "input_rows": len(raw), "output_rows": len(snapped), "rejected_rows": len(rejected),
            "snap_dist_median_m": float(d.median()) if len(d) else float("nan"),
            "snap_dist_p95_m": float(d.quantile(0.95)) if len(d) else float("nan"),
            "low_confidence_share": float((snapped["snap_quality"] != "good").mean())
            if len(d) else float("nan"),
        }
        logger.info("Ingestion stats: %s", {k: round(v, 3) for k, v in stats.items()})
        return stats

    def _write(self, snapped: gpd.GeoDataFrame, rejected: pd.DataFrame) -> None:
        self.paths.ensure()
        snapped.to_parquet(self.paths.snapped_collisions, index=False)
        rejected.drop(columns=[c for c in ("geometry", "raw_geometry") if c in rejected],
                      errors="ignore").to_csv(self.paths.rejected_records, index=False)
        logger.info("Wrote %s and %s", self.paths.snapped_collisions, self.paths.rejected_records)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Iterable[str]] = None) -> IngestionResult:
    parser = argparse.ArgumentParser(description="Ingest, reproject and road-snap LA collisions.")
    parser.add_argument("-i", "--input", type=Path, default=settings.paths.raw_collisions)
    parser.add_argument("--graphml", type=Path, default=None,
                        help="Use this GraphML instead of the cache / OSM download.")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    pipeline = IngestionPipeline()
    graph = None
    if args.force_download:
        graph = pipeline.network_loader.load(force_download=True)
    return pipeline.run(args.input, graph=graph, graphml=args.graphml)


if __name__ == "__main__":
    main()
