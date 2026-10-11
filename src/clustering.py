"""
DBSCAN hotspot detection: dangerous intersections and road corridors.

Why DBSCAN rather than K-Means
------------------------------
K-Means forces every crash into one of *k* round blobs. Crash hotspots are
neither: a dangerous junction is a tight knot, a dangerous arterial is a long
thin line, and most crashes belong to no hotspot at all. DBSCAN grows clusters
by chaining points that each have ``min_samples`` neighbours within ``eps``
metres, so it follows a road of any length and shape and labels isolated
crashes as noise (-1).

Two-scale tuning
----------------
One global (eps, min_samples) cannot find both hotspot shapes: a radius small
enough to separate junction knots shatters corridors into noise, and one large
enough to chain a corridor merges neighbouring junctions and background. So:

* **Stage 1 – intersections.** Grid-search a tight eps on all crashes.
* **Stage 2 – corridors.** Re-run on the stage-1 noise with a wider eps,
  scoring only clusters that are *corridor-like*: long (≥ 4·eps) and either
  straight **and narrow** (PCA linearity ≥ 0.9, width ≤ 60 m) **or** following
  one named road across ≥ 3 segments (curving roads), **and** at least
  ``corridor_min_local_lift``× denser than a 500 m ring around them. Width and
  lift reject chance chains of background crashes that a wide eps strings
  together: those can look 90 % linear over a kilometre yet be hundreds of
  metres wide and only marginally denser than their surroundings.

Each stage builds one sparse radius graph at its largest eps and reuses it for
every grid point, so the search is cheap. Selection maximises

    capture_score = coverage × ln(PAI)

where PAI (Prediction Accuracy Index, standard in crime/crash hotspot
analysis) = share of crashes inside hotspots ÷ share of city area they cover,
and area is measured as occupied 25 m raster cells so it follows the road
footprint rather than a convex hull. Maximising PAI alone keeps shrinking eps
until only the tightest knots survive (on planted test data it recovered half
the intersections and fragmented every corridor); coverage alone would chain
the whole city. Constraints: PAI ≥ ``min_pai``, minimum coverage and cluster
count, and no cluster holding more than ``max_largest_cluster_share`` of all
crashes. The k-distance knee is reported as a sanity reference.

Each cluster is then typed **intersection / corridor / area** from its shape
(PCA linearity and length), its dominant named road, and its median distance
to a junction, and ranked by insured loss.

Leakage note for modelling: cluster membership derived from all crashes must
not be used as a feature when evaluating on held-out crashes; refit
``HotspotClusteringPipeline`` inside each spatial CV fold.

Usage::

    python -m src.clustering
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy import sparse
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from config import settings
from config.config import ClusteringConfig, CRSConfig, PathConfig, StudyAreaConfig
from src.utils import get_logger, timed

logger = get_logger(__name__)

NOISE = -1


# --------------------------------------------------------------------------- #
# k-distance analysis
# --------------------------------------------------------------------------- #
@dataclass
class KDistanceResult:
    k: int
    distances: np.ndarray           # sorted ascending
    knee_eps_m: float
    knee_index: int


class KDistanceAnalyzer:
    """Estimate eps from the elbow of the sorted k-nearest-neighbour distance curve."""

    def __init__(self, k: int = settings.clustering.knn_k) -> None:
        self.k = k

    def fit(self, X: np.ndarray) -> KDistanceResult:
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)   # +1: the point itself
        d = np.sort(nn.kneighbors(X)[0][:, -1])
        # Kneedle: point furthest below the chord of the normalised curve,
        # ignoring the top 2 % so a few isolated crashes don't flatten it.
        cut = max(int(len(d) * 0.98), 2)
        y = d[:cut]
        x = np.linspace(0, 1, cut)
        yn = (y - y[0]) / max(y[-1] - y[0], 1e-9)
        idx = int(np.argmax(x - yn))
        return KDistanceResult(self.k, d, float(d[idx]), idx)

    @staticmethod
    def plot(res: KDistanceResult, chosen: Dict[str, float], path: Path) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:                                     # pragma: no cover
            return None
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(len(res.distances)), res.distances, lw=1.2, color="#1f5f99")
        ax.axhline(res.knee_eps_m, ls="--", color="#888", lw=1,
                   label=f"knee ≈ {res.knee_eps_m:.0f} m")
        for (stage, eps), col in zip(chosen.items(), ("#c2410c", "#1d4ed8")):
            ax.axhline(eps, ls="-", color=col, lw=1.2, label=f"{stage} eps = {eps:.0f} m")
        ax.set_yscale("log")
        ax.set_xlabel("collisions sorted by k-distance")
        ax.set_ylabel(f"distance to {res.k}-th neighbour (m)")
        ax.legend(frameon=False)
        ax.set_title("k-distance curve")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# --------------------------------------------------------------------------- #
# Grid search
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageSpec:
    """Search space and constraints for one DBSCAN scale."""

    name: str                       # "intersection" | "corridor"
    eps_grid_m: Tuple[float, ...]
    min_samples_grid: Tuple[int, ...]
    min_coverage: float             # share of ALL crashes that must be captured
    min_clusters: int
    corridor_only: bool = False     # score only corridor-like clusters


@dataclass
class TuningResult:
    stage: str
    eps_m: float
    min_samples: int
    labels: np.ndarray              # labels for the points the stage ran on
    table: pd.DataFrame
    feasible: bool


class DBSCANTuner:
    """Grid-search eps × min_samples for one stage, selecting by PAI under constraints."""

    def __init__(self, cfg: ClusteringConfig = settings.clustering,
                 study_area_m2: Optional[float] = None) -> None:
        self.cfg = cfg
        self.study_area_m2 = study_area_m2 or land_area_m2()

    def tune(self, X: np.ndarray, spec: StageSpec, n_total: int,
             road: Optional[np.ndarray] = None, segment: Optional[np.ndarray] = None,
             sample_weight: Optional[np.ndarray] = None) -> TuningResult:
        """
        Parameters
        ----------
        X: (n, 2) projected coordinates of the points this stage clusters.
        n_total: crash count of the whole dataset (coverage/PAI denominators).
        road, segment: per-point road name and segment id codes, used to
            recognise curving corridors (optional).
        """
        eps_grid = sorted(set(float(e) for e in spec.eps_grid_m))
        with timed(logger, f"[{spec.name}] radius graph at eps={max(eps_grid):.0f} m"):
            graph = NearestNeighbors(radius=max(eps_grid)).fit(X) \
                .radius_neighbors_graph(X, mode="distance", sort_results=True)
        cells = self._cells(X)

        rows, labels_by = [], {}
        with timed(logger, f"[{spec.name}] DBSCAN grid "
                           f"({len(eps_grid)}×{len(spec.min_samples_grid)})"):
            for eps in eps_grid:
                g = graph.copy()
                g.data[g.data > eps] = 0.0                     # drop edges beyond eps …
                g.eliminate_zeros()
                g = (g + sparse.identity(g.shape[0], format="csr") * 1e-9).tocsr()  # … keep self
                for ms in spec.min_samples_grid:
                    lab = self._drop_small(
                        DBSCAN(eps=eps, min_samples=ms, metric="precomputed")
                        .fit_predict(g, sample_weight=sample_weight))
                    keep = self._corridor_like(X, lab, eps, road, segment) \
                        if spec.corridor_only else None
                    rows.append({"stage": spec.name, "eps_m": eps, "min_samples": ms,
                                 **self.score(lab, cells, n_total, keep)})
                    labels_by[(eps, ms)] = lab
        table = pd.DataFrame(rows)
        table["capture_score"] = (table["coverage"]
                                  * np.log(table["pai"].clip(lower=1.0))).round(4)
        table["feasible"] = ((table["coverage"] >= spec.min_coverage)
                             & (table["pai"] >= self.cfg.min_pai)
                             & (table["n_clusters"] >= spec.min_clusters)
                             & (table["largest_cluster_share"]
                                <= self.cfg.max_largest_cluster_share))

        pool = table[table["feasible"]]
        feasible = not pool.empty
        if not feasible:
            logger.warning("[%s] no parameter set met all constraints; relaxing to the best "
                           "capture score under the mega-cluster limit.", spec.name)
            pool = table[table["largest_cluster_share"] <= self.cfg.max_largest_cluster_share]
            pool = pool if not pool.empty else table
        best = pool.sort_values(["capture_score", "pai"], ascending=False).iloc[0]
        eps, ms = float(best["eps_m"]), int(best["min_samples"])
        labels = labels_by[(eps, ms)]
        if spec.corridor_only:   # keep only the corridor-like clusters from this stage
            keep = self._corridor_like(X, labels, eps, road, segment)
            labels = _relabel(np.where(np.isin(labels, np.flatnonzero(keep)), labels, NOISE))
        logger.info("[%s] selected eps=%.0f m, min_samples=%d → %d clusters, coverage %.1f%%, "
                    "PAI %.0f", spec.name, eps, ms, best["n_clusters"],
                    100 * best["coverage"], best["pai"])
        if eps in (min(eps_grid), max(eps_grid)) or \
                ms in (min(spec.min_samples_grid), max(spec.min_samples_grid)):
            logger.info("[%s] selection is on the edge of the search grid; consider "
                        "widening it for this dataset.", spec.name)
        return TuningResult(spec.name, eps, ms, labels,
                            table.sort_values(["feasible", "capture_score"], ascending=False)
                            .reset_index(drop=True), feasible)

    def score(self, labels: np.ndarray, cells: np.ndarray, n_total: int,
              keep: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Hotspot metrics; ``keep`` (bool per cluster id) restricts to selected clusters."""
        lab = labels
        if keep is not None:
            lab = np.where((labels != NOISE) & keep[np.maximum(labels, 0)], labels, NOISE) \
                if keep.size else np.full_like(labels, NOISE)
        clustered = lab != NOISE
        ids = np.unique(lab[clustered])
        coverage = clustered.sum() / n_total
        if ids.size:
            sizes = np.bincount(lab[clustered])
            area = np.unique(cells[clustered]).size * self.cfg.footprint_cell_m ** 2
            pai = coverage / (area / self.study_area_m2)
            largest = sizes.max() / n_total
            median_size = float(np.median(sizes[sizes > 0]))
        else:
            area = pai = largest = median_size = 0.0
        return {"n_clusters": int(ids.size), "coverage": round(float(coverage), 4),
                "hotspot_area_km2": round(area / 1e6, 3), "pai": round(float(pai), 2),
                "largest_cluster_share": round(float(largest), 4),
                "median_cluster_size": median_size}

    def _corridor_like(self, X: np.ndarray, labels: np.ndarray, eps: float,
                       road: Optional[np.ndarray], segment: Optional[np.ndarray]) -> np.ndarray:
        """Bool per cluster id: straight-and-long, or one named road over ≥ N segments."""
        n_cl = int(labels.max()) + 1 if (labels != NOISE).any() else 0
        if n_cl == 0:
            return np.zeros(0, dtype=bool)
        shape = cluster_shapes(X, labels, n_cl)
        c = self.cfg
        straight = ((shape["linearity"] >= c.corridor_min_linearity)
                    & (shape["width_m"] <= c.corridor_max_width_m)
                    & (shape["length_m"] >= c.corridor_min_length_eps * eps))
        if road is None or segment is None:
            return straight.to_numpy()
        m = labels != NOISE
        df = pd.DataFrame({"lab": labels[m], "road": road[m], "seg": segment[m]})
        df = df[df["road"] >= 0]                                # -1 = unnamed
        share = df.groupby(["lab", "road"]).size().groupby(level=0).max() \
            / df.groupby("lab").size()
        segs = df.groupby("lab")["seg"].nunique()
        on_road = ((share.reindex(range(n_cl)).fillna(0) >= c.corridor_min_road_share)
                   & (segs.reindex(range(n_cl)).fillna(0) >= c.corridor_min_segments)
                   & (shape["length_m"] >= c.corridor_min_length_eps * eps))
        cand = (straight | on_road).to_numpy()
        return cand & (self.local_lift(X, labels, eps, np.flatnonzero(cand), n_cl)
                       >= c.corridor_min_local_lift)

    def local_lift(self, X: np.ndarray, labels: np.ndarray, eps: float,
                   ids: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Crash density inside each cluster's footprint (hull ⊕ one 25 m raster
        cell, i.e. road scale) divided by density in a ``local_ring_m`` ring
        around it. Returns 0 for ids not asked.
        """
        lift = np.zeros(n_clusters)
        if ids.size == 0:
            return lift
        pad = self.cfg.footprint_cell_m
        hulls = [shapely.convex_hull(shapely.multipoints(X[labels == i])).buffer(pad)
                 for i in ids]
        rings = [shapely.difference(h.buffer(self.cfg.local_ring_m), h) for h in hulls]
        tree = shapely.STRtree(shapely.points(X))
        ring_hits = np.bincount(tree.query(rings, predicate="contains")[0], minlength=len(ids))
        n_in = np.bincount(labels[labels != NOISE], minlength=n_clusters)[ids]
        a_in = shapely.area(hulls)
        a_ring = shapely.area(rings)
        ring_density = np.maximum(ring_hits, 1) / a_ring     # ≥1 crash avoids ∞ lift
        lift[ids] = (n_in / a_in) / ring_density
        return lift

    def _drop_small(self, labels: np.ndarray) -> np.ndarray:
        """Return clusters below ``min_hotspot_crashes`` to noise and relabel."""
        if not (labels != NOISE).any():
            return labels
        sizes = np.bincount(labels[labels != NOISE])
        small = np.flatnonzero(sizes < self.cfg.min_hotspot_crashes)
        return _relabel(np.where(np.isin(labels, small), NOISE, labels))

    def _cells(self, X: np.ndarray) -> np.ndarray:
        ij = np.floor(X / self.cfg.footprint_cell_m).astype(np.int64)
        return ij[:, 0] * 10_000_000 + ij[:, 1]


def cluster_shapes(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> pd.DataFrame:
    """
    Vectorised per-cluster PCA. λ1 ≥ λ2 are the variances along the major and
    minor axes; ``linearity = 1 − λ2/λ1`` (≈1 for a line) and
    ``length_m = √(12·λ1)`` (exact for points spread evenly along a segment);
    ``width_m = √(12·λ2)`` is the same measure across the major axis.
    """
    m = labels != NOISE
    lab, x, y = labels[m], X[m, 0], X[m, 1]
    n = np.bincount(lab, minlength=n_clusters).astype(float)
    nz = np.maximum(n, 1)
    mx, my = np.bincount(lab, x, n_clusters) / nz, np.bincount(lab, y, n_clusters) / nz
    dx, dy = x - mx[lab], y - my[lab]
    sxx = np.bincount(lab, dx * dx, n_clusters) / nz
    syy = np.bincount(lab, dy * dy, n_clusters) / nz
    sxy = np.bincount(lab, dx * dy, n_clusters) / nz
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    lin = np.where(l1 > 0, 1 - l2 / np.where(l1 > 0, l1, 1), 0.0)
    return pd.DataFrame({"n": n, "cx": mx, "cy": my, "length_m": np.sqrt(12 * l1),
                         "width_m": np.sqrt(12 * l2), "linearity": lin})


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Make cluster ids contiguous 0..k-1, keeping NOISE."""
    out = np.full_like(labels, NOISE)
    m = labels != NOISE
    if m.any():
        _, out[m] = np.unique(labels[m], return_inverse=True)
    return out


class MultiScaleDBSCAN:
    """Stage 1 (intersections) on all points, stage 2 (corridors) on stage-1 noise."""

    def __init__(self, cfg: ClusteringConfig = settings.clustering,
                 study_area_m2: Optional[float] = None) -> None:
        self.cfg = cfg
        self.tuner = DBSCANTuner(cfg, study_area_m2)
        self.stages = (
            StageSpec("intersection", cfg.intersection_eps_grid_m,
                      cfg.intersection_min_samples_grid, cfg.intersection_min_coverage,
                      cfg.intersection_min_clusters),
            StageSpec("corridor", cfg.corridor_eps_grid_m, cfg.corridor_min_samples_grid,
                      cfg.corridor_min_coverage, cfg.corridor_min_clusters, corridor_only=True),
        )

    def fit(self, X: np.ndarray, road: Optional[np.ndarray] = None,
            segment: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, np.ndarray, List[TuningResult]]:
        """Return (labels, stage-of-point, per-stage tuning results)."""
        n = len(X)
        labels = np.full(n, NOISE)
        stage_of = np.full(n, "noise", dtype=object)
        results: List[TuningResult] = []
        remaining = np.arange(n)
        offset = 0
        for spec in self.stages:
            if remaining.size == 0:
                break
            sub = lambda a: None if a is None else a[remaining]  # noqa: E731
            r = self.tuner.tune(X[remaining], spec, n, sub(road), sub(segment),
                                sub(sample_weight))
            hit = r.labels != NOISE
            labels[remaining[hit]] = r.labels[hit] + offset
            stage_of[remaining[hit]] = spec.name
            offset = labels.max() + 1 if (labels != NOISE).any() else 0
            results.append(r)
            remaining = remaining[~hit]
        return labels, stage_of, results


def land_area_m2(area_cfg: StudyAreaConfig = settings.study_area,
                 crs_cfg: CRSConfig = settings.crs) -> float:
    """Area of the study-area land polygon in projected square metres."""
    t = Transformer.from_crs(crs_cfg.geographic, crs_cfg.projected, always_xy=True)
    x, y = t.transform(*zip(*area_cfg.land_polygon))
    return Polygon(zip(x, y)).area


# --------------------------------------------------------------------------- #
# Cluster profiling & typing
# --------------------------------------------------------------------------- #
class ClusterProfiler:
    """
    Describe each cluster: shape, dominant road, junction proximity, severity
    and insured loss; assign ``cluster_type`` and a loss-based rank.

    Shape comes from ``cluster_shapes`` (PCA linearity and length).
    """

    def __init__(self, stage_eps: Dict[str, float],
                 cfg: ClusteringConfig = settings.clustering) -> None:
        self.stage_eps, self.cfg = stage_eps, cfg

    def profile(self, gdf: gpd.GeoDataFrame, labels: np.ndarray,
                stage_of: np.ndarray) -> gpd.GeoDataFrame:
        X = np.column_stack([gdf.geometry.x, gdf.geometry.y])
        n_cl = int(labels.max()) + 1 if (labels != NOISE).any() else 0
        shapes = cluster_shapes(X, labels, n_cl)
        lifts = np.zeros(n_cl)
        tuner = DBSCANTuner(self.cfg, study_area_m2=1.0)
        for stage, eps in self.stage_eps.items():
            ids = np.unique(labels[(stage_of == stage) & (labels != NOISE)])
            lifts += tuner.local_lift(X, labels, eps, ids, n_cl)
        rows, hulls = [], []
        for cid in range(n_cl):
            m = labels == cid
            sub = gdf.loc[m]
            pts = X[m]
            stage = str(stage_of[m][0])
            linearity, length = shapes.at[cid, "linearity"], shapes.at[cid, "length_m"]
            roads = sub["road_name"].dropna() if "road_name" in sub else pd.Series(dtype=str)
            dom_road = roads.mode().iloc[0] if len(roads) else None
            road_share = float((roads == dom_road).mean()) if len(roads) else 0.0
            d_int = float(sub["dist_to_intersection_m"].median()) \
                if "dist_to_intersection_m" in sub else np.nan
            rows.append({
                "cluster_id": int(cid), "stage": stage, "eps_m": self.stage_eps[stage],
                "local_lift": round(float(lifts[cid]), 2),
                "n_collisions": int(m.sum()),
                "centroid_x": pts[:, 0].mean(), "centroid_y": pts[:, 1].mean(),
                "length_m": round(length, 1), "width_m": round(float(shapes.at[cid, "width_m"]), 1),
                "linearity": round(float(linearity), 3),
                "dominant_road": dom_road, "dominant_road_share": round(road_share, 3),
                "n_road_segments": int(sub[["edge_u", "edge_v"]].apply(
                    lambda r: tuple(sorted(r)), axis=1).nunique()) if "edge_u" in sub else 0,
                "median_dist_to_intersection_m": round(d_int, 1),
                "mean_severity_code": round(float(sub["severity_code"].mean()), 3)
                if "severity_code" in sub else np.nan,
                "severe_fatal_share": round(float((sub["severity_code"] >= 2).mean()), 3)
                if "severity_code" in sub else np.nan,
                "total_claims_usd": round(float(sub["claim_amount_usd"].sum()), 2)
                if "claim_amount_usd" in sub else np.nan,
                "mean_claim_usd": round(float(sub["claim_amount_usd"].mean()), 2)
                if "claim_amount_usd" in sub else np.nan,
            })
            hulls.append(shapely.convex_hull(shapely.multipoints(pts)).buffer(15.0))

        cols = ["cluster_id", "n_collisions", "cluster_type"]
        if not rows:
            return gpd.GeoDataFrame(columns=cols, geometry=[], crs=gdf.crs)
        summary = gpd.GeoDataFrame(rows, geometry=hulls, crs=gdf.crs)
        summary["cluster_type"] = self._classify(summary)
        summary["claims_per_km_road"] = np.where(
            summary["cluster_type"] == "corridor",
            summary["total_claims_usd"] / np.maximum(summary["length_m"] / 1000, 0.1), np.nan)
        summary["loss_rank"] = summary["total_claims_usd"].rank(ascending=False,
                                                                method="first").astype(int)
        return summary.sort_values("loss_rank").reset_index(drop=True)

    def _classify(self, s: pd.DataFrame) -> np.ndarray:
        c, eps = self.cfg, s["eps_m"]
        is_int = ((s["median_dist_to_intersection_m"] <= c.intersection_max_dist_m)
                  & (s["length_m"] <= c.intersection_max_length_m))
        long_ = s["length_m"] >= c.corridor_min_length_eps * eps
        is_cor = long_ & (s["local_lift"] >= c.corridor_min_local_lift) & (
            ((s["linearity"] >= c.corridor_min_linearity) & (s["width_m"] <= c.corridor_max_width_m))
            | ((s["dominant_road_share"] >= c.corridor_min_road_share)
               & (s["n_road_segments"] >= c.corridor_min_segments)))
        return np.select([is_int, is_cor], ["intersection", "corridor"], "area")


# --------------------------------------------------------------------------- #
# Ground-truth evaluation (synthetic data only)
# --------------------------------------------------------------------------- #
class HotspotRecoveryEvaluator:
    """
    Score clustering against hotspots planted by ``NetworkHotspotPlanter``.

    *Detectable* hotspots: ≥ ``min_crashes`` crashes, and for corridors also
    ≥ ``corridor_min_per_100m`` crashes per 100 m (a 3 km road with 12 crashes
    is not a hotspot any method should find). A hotspot is *recovered* when
    ≥ 50 % of its crashes sit in clusters whose majority is that hotspot —
    a corridor split into a few pure fragments still counts, which is reported
    separately as ``*_fragments``.
    """

    def __init__(self, min_crashes: int = 10, corridor_min_per_100m: float = 1.5) -> None:
        self.min_crashes, self.min_density = min_crashes, corridor_min_per_100m

    def evaluate(self, gt_id: pd.Series, gt_type: pd.Series, labels: np.ndarray,
                 cluster_types: Dict[int, str], X: np.ndarray) -> Dict[str, float]:
        df = pd.DataFrame({"hid": gt_id.to_numpy(), "kind": gt_type.to_numpy(),
                           "lab": labels, "x": X[:, 0], "y": X[:, 1]})
        clustered = df[df["lab"] != NOISE]
        majority = clustered.groupby("lab")["hid"].agg(lambda s: s.value_counts().index[0])
        df["pure_hit"] = df["lab"].map(majority).eq(df["hid"]) & (df["lab"] != NOISE)

        out: Dict[str, float] = {}
        type_hits: List[bool] = []
        for kind in ("intersection", "corridor"):
            sub = df[df["kind"] == kind]
            n = sub.groupby("hid").size()
            ok = n >= self.min_crashes
            if kind == "corridor" and len(n):
                length = sub.groupby("hid")[["x", "y"]].apply(
                    lambda g: np.sqrt(12 * np.linalg.eigvalsh(np.cov(g.to_numpy().T)).max())
                    if len(g) > 2 else 0.0)
                ok &= n / np.maximum(length / 100.0, 1e-9) >= self.min_density
            det = sub[sub["hid"].isin(n[ok].index)]
            share = det.groupby("hid")["pure_hit"].mean()
            rec = share[share >= 0.5].index
            frags = det[det["hid"].isin(rec) & det["pure_hit"]].groupby("hid")["lab"].nunique()
            out[f"{kind}_detectable"] = int(ok.sum())
            out[f"{kind}_recall"] = round(float((share >= 0.5).mean()), 3) if len(share) else np.nan
            out[f"{kind}_crash_capture"] = round(float(share.mean()), 3) if len(share) else np.nan
            out[f"{kind}_fragments"] = round(float(frags.mean()), 2) if len(frags) else np.nan
            for lab in det.loc[det["hid"].isin(rec) & det["pure_hit"], "lab"].unique():
                type_hits.append(cluster_types.get(int(lab)) == kind)

        out["cluster_precision"] = round(float((majority != "none").mean()), 3) \
            if len(majority) else 0.0
        out["type_accuracy"] = round(float(np.mean(type_hits)), 3) if type_hits else np.nan
        out["background_in_hotspots"] = round(float(
            (df.loc[df["hid"] == "none", "lab"] != NOISE).mean()), 4)
        return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class ClusteringResult:
    data: gpd.GeoDataFrame                 # collisions + cluster_id / cluster_type
    clusters: gpd.GeoDataFrame             # one row per hotspot
    stages: List[TuningResult]
    knee: KDistanceResult
    evaluation: Dict[str, float] = field(default_factory=dict)

    @property
    def params(self) -> Dict[str, Tuple[float, int]]:
        return {r.stage: (r.eps_m, r.min_samples) for r in self.stages}


class HotspotClusteringPipeline:
    """Tune DBSCAN on snapped UTM coordinates, profile clusters, evaluate, persist."""

    def __init__(self, cfg: ClusteringConfig = settings.clustering,
                 paths: PathConfig = settings.paths) -> None:
        self.cfg, self.paths = cfg, paths

    def run(self, gdf: gpd.GeoDataFrame, write: bool = True,
            weight_col: Optional[str] = None) -> ClusteringResult:
        if gdf.crs is None or not gdf.crs.is_projected:
            raise ValueError("Clustering needs projected coordinates in metres (EPSG:32611).")
        gdf = gdf.reset_index(drop=True)
        X = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
        w = gdf[weight_col].to_numpy(float) if weight_col else None

        road = pd.factorize(gdf["road_name"])[0] if "road_name" in gdf else None   # NaN → -1
        segment = pd.factorize(pd.Series(list(zip(
            np.minimum(gdf["edge_u"], gdf["edge_v"]),
            np.maximum(gdf["edge_u"], gdf["edge_v"])))))[0] if "edge_u" in gdf else None

        knee = KDistanceAnalyzer(self.cfg.knn_k).fit(X)
        logger.info("k-distance knee (k=%d): %.1f m (reference only)", knee.k, knee.knee_eps_m)
        labels, stage_of, stages = MultiScaleDBSCAN(self.cfg).fit(X, road, segment, w)
        stage_eps = {r.stage: r.eps_m for r in stages}
        clusters = ClusterProfiler(stage_eps, self.cfg).profile(gdf, labels, stage_of)
        types = dict(zip(clusters["cluster_id"], clusters["cluster_type"]))
        ranks = dict(zip(clusters["cluster_id"], clusters["loss_rank"]))

        data = gdf.copy()
        data["cluster_id"] = labels
        data["cluster_stage"] = stage_of.astype(str)
        data["cluster_type"] = pd.Series(labels).map(types).fillna("noise").to_numpy()
        data["in_hotspot"] = (labels != NOISE).astype(int)
        data["hotspot_loss_rank"] = pd.Series(labels).map(ranks).astype("Int64").to_numpy()
        logger.info("Cluster types: %s", clusters["cluster_type"].value_counts().to_dict())

        evaluation: Dict[str, float] = {}
        if "gt_hotspot_id" in gdf:
            evaluation = HotspotRecoveryEvaluator().evaluate(
                gdf["gt_hotspot_id"].astype(str), gdf["gt_hotspot_type"].astype(str),
                labels, types, X)
            logger.info("Ground-truth recovery: %s", evaluation)

        result = ClusteringResult(data, clusters, stages, knee, evaluation)
        if write:
            self._write(result)
        return result

    def _write(self, r: ClusteringResult) -> None:
        self.paths.ensure()
        r.data.drop(columns=[c for c in ("raw_geometry",) if c in r.data]) \
            .to_parquet(self.paths.clustered, index=False)
        r.clusters.to_parquet(self.paths.cluster_summary, index=False)
        rd = self.paths.reports_dir
        pd.concat([t.table for t in r.stages]).to_csv(rd / "dbscan_tuning_grid.csv", index=False)
        if r.evaluation:
            pd.Series(r.evaluation).to_csv(rd / "dbscan_groundtruth_eval.csv", header=["value"])
        KDistanceAnalyzer.plot(r.knee, {t.stage: t.eps_m for t in r.stages},
                               rd / "k_distance_curve.png")
        self.plot_map(r, rd / "hotspot_map.png")
        logger.info("Wrote %s, %s and reports to %s", self.paths.clustered,
                    self.paths.cluster_summary, rd)

    @staticmethod
    def plot_map(r: ClusteringResult, path: Path) -> Optional[Path]:
        """Static overview: noise in grey, hotspots coloured by type."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:                                     # pragma: no cover
            return None
        colours = {"intersection": "#c2410c", "corridor": "#1d4ed8", "area": "#7c3aed"}
        d = r.data
        fig, ax = plt.subplots(figsize=(8, 9))
        noise = d["cluster_id"] == NOISE
        ax.scatter(d.geometry.x[noise], d.geometry.y[noise], s=0.4, c="#c8c8c8", lw=0)
        for kind, col in colours.items():
            m = d["cluster_type"] == kind
            ax.scatter(d.geometry.x[m], d.geometry.y[m], s=2.5, c=col, lw=0,
                       label=f"{kind} ({(r.clusters['cluster_type'] == kind).sum()})")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.legend(loc="lower left", frameon=False, markerscale=4)
        ax.set_title("Two-scale DBSCAN hotspots — " + "; ".join(
            f"{k}: eps {e:.0f} m, min_samples {m}" for k, (e, m) in r.params.items()),
            fontsize=9)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Iterable[str]] = None) -> ClusteringResult:
    parser = argparse.ArgumentParser(description="DBSCAN crash hotspot detection.")
    parser.add_argument("-i", "--input", type=Path, default=settings.paths.features)
    parser.add_argument("--weight-col", default=None,
                        help="Optional per-crash weight (e.g. severity) for DBSCAN.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return HotspotClusteringPipeline().run(gpd.read_parquet(args.input),
                                           weight_col=args.weight_col)


if __name__ == "__main__":
    main()
