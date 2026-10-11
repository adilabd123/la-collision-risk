"""
Severity-risk model training: High_Risk classifier, thresholds and 4-tier spatial risk.

Steps
-----
1. **Target** – ``High_Risk = 1`` for severe-injury or fatal crashes
   (``severity_code >= 2``, ≈ 11.5 % of records), 0 otherwise.
2. **Spatial split** – crashes are grouped into 2 km blocks and whole blocks
   go to train or test, so the model is tested on places it never saw.
   Neighbouring crashes share road, lighting and hotspot context; a random
   split would leak that and inflate every score.
3. **Hotspots on train only** – the two-scale DBSCAN is refit on the training
   split; test crashes are assigned with ``HotspotMatcher`` (DBSCAN's
   core-point rule), exactly as the API does.
4. **Models** – Logistic Regression, Random Forest, XGBoost, each an imblearn
   pipeline: impute → hotspot target encoding → **SMOTENC (training folds
   only)** → encode → classifier. SMOTENC is SMOTE for mixed data: it
   interpolates numeric columns and takes the majority category among
   neighbours instead of inventing fractional one-hot values. It lifts the
   minority to 1:2, then class weights (``class_weight='balanced'`` for RF,
   the equivalent ``scale_pos_weight`` for XGBoost) close the rest.
5. **Threshold** – chosen per model on out-of-fold predictions by maximising
   F2 (recall counts 4× precision). The fixed 0.20 and the default 0.50 are
   reported alongside.
6. **Evaluation** – PR-AUC (headline: honest under imbalance), ROC-AUC, Brier,
   confusion matrices at all three thresholds, on the held-out blocks.
7. **4-tier risk** – crash-level tiers from out-of-fold probability quantiles
   (50/80/95 %), and a spatial tier per road segment from expected severe
   crashes per year (frequency × shrunk severity propensity).

Usage::

    python -m src.train                              # underwriting feature set
    python -m src.train --feature-set claims_triage  # adds post-crash facts
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             fbeta_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from config import settings
from config.config import ModelConfig, PathConfig
from src.clustering import HotspotClusteringPipeline
from src.graph_utils import edges_gdf
from src.inference import HotspotMatcher, ModelFeatureBuilder, add_derived, assign_tier
from src.ingestion import RoadNetworkLoader
from src.utils import get_logger, timed

logger = get_logger(__name__)

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")
MODEL_COLOURS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834",
                 "xgboost": "#1baf7a"}          # validated categorical slots 1–3
TIER_COLOURS = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")   # validated ordinal ramp


# --------------------------------------------------------------------------- #
# Target & split
# --------------------------------------------------------------------------- #
def make_target(df: pd.DataFrame, cfg: ModelConfig = settings.model) -> pd.Series:
    """High_Risk = 1 for severe injury or fatal crashes."""
    return (df["severity_code"] >= cfg.high_risk_min_severity).astype(int).rename("High_Risk")


class SpatialSplitter:
    """Group crashes into square blocks so train/test and CV folds never share a block."""

    def __init__(self, cfg: ModelConfig = settings.model) -> None:
        self.cfg = cfg

    def groups(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        b = self.cfg.spatial_block_size_m
        ix = np.floor(gdf.geometry.x.to_numpy() / b).astype(np.int64)
        iy = np.floor(gdf.geometry.y.to_numpy() / b).astype(np.int64)
        return ix * 100_000 + iy

    def train_test(self, gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
        gss = GroupShuffleSplit(n_splits=1, test_size=self.cfg.test_size,
                                random_state=self.cfg.random_seed)
        return next(gss.split(gdf, groups=self.groups(gdf)))

    def cv(self) -> GroupKFold:
        return GroupKFold(n_splits=self.cfg.spatial_cv_folds)


# --------------------------------------------------------------------------- #
# Model pipelines
# --------------------------------------------------------------------------- #
def build_pipeline(name: str, builder: ModelFeatureBuilder, prevalence: float,
                   cfg: ModelConfig = settings.model,
                   oversampler: Optional[str] = None) -> ImbPipeline:
    """
    impute + hotspot target encoding → (oversample) → classifier.

    ``oversampler``:
      * ``smote``    – SMOTE on the one-hot-encoded matrix (default). Category
                       dummies are interpolated, so minority category mixes
                       are preserved in expectation.
      * ``smotenc``  – SMOTENC on raw categories. Gives each synthetic row the
                       most frequent category among its neighbours, which
                       pulls minority rows toward common categories.
      * ``none``     – no resampling; class weights only.
    """
    oversampler = oversampler or cfg.oversampler
    num, cat = builder.numeric_cols, builder.categorical_cols
    seed, s = cfg.random_seed, cfg.smote_sampling_strategy
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", SimpleImputer(strategy="constant", fill_value="missing"), cat),
        # Hotspot severe-crash rate. sklearn's TargetEncoder cross-fits inside
        # fit_transform, so a training crash's own label never feeds its value.
        ("te", TargetEncoder(target_type="binary",
                             cv=KFold(5, shuffle=True, random_state=seed)), ["cluster_id"]),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    encode = ColumnTransformer([
        ("num", StandardScaler() if name == "logistic_regression" else "passthrough",
         num + ["cluster_id"]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                              sparse_output=False), cat),
    ])

    if oversampler == "smote":
        steps = [("prep", prep), ("encode", encode),
                 ("smote", SMOTE(sampling_strategy=s, k_neighbors=cfg.smote_k_neighbors,
                                 random_state=seed))]
        pos_ratio = s                                  # minority:majority after SMOTE
    elif oversampler == "smotenc":
        steps = [("prep", prep),
                 ("smote", SMOTENC(categorical_features=cat, sampling_strategy=s,
                                   k_neighbors=cfg.smote_k_neighbors, random_state=seed)),
                 ("encode", encode)]
        pos_ratio = s
    elif oversampler == "none":
        steps = [("prep", prep), ("encode", encode)]
        pos_ratio = prevalence / (1 - prevalence)
    else:
        raise ValueError(f"Unknown oversampler {oversampler!r}")

    if name == "logistic_regression":
        # Baseline: oversampling only (balanced weights if not oversampling).
        clf = LogisticRegression(**cfg.lr_params, random_state=seed,
                                 class_weight="balanced" if oversampler == "none" else None)
    elif name == "random_forest":
        clf = RandomForestClassifier(**cfg.rf_params, class_weight="balanced",
                                     random_state=seed)
    elif name == "xgboost":
        # XGBoost has no class_weight; scale_pos_weight = majority/minority at
        # fit time is the same 'balanced' correction.
        clf = XGBClassifier(**cfg.xgb_params, scale_pos_weight=1.0 / pos_ratio,
                            random_state=seed)
    else:
        raise ValueError(name)
    return ImbPipeline(steps + [("clf", clf)])


# --------------------------------------------------------------------------- #
# Thresholds, metrics, tiers
# --------------------------------------------------------------------------- #
def optimise_threshold(y: np.ndarray, p: np.ndarray,
                       cfg: ModelConfig = settings.model) -> Tuple[float, pd.DataFrame]:
    """
    Threshold maximising F-beta on out-of-fold predictions, among thresholds
    that flag at most ``max_flag_share`` of crashes.

    The cap matters when signal is weak: F2 rewards recall so heavily that
    "flag everything" (recall 1, precision = base rate) can win outright,
    which is useless operationally.
    """
    rows = []
    for t in cfg.threshold_grid:
        pred = p >= t
        rows.append({"threshold": t, "flagged_share": float(pred.mean()),
                     "precision": precision_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0),
                     f"f{cfg.threshold_beta:g}": fbeta_score(y, pred, beta=cfg.threshold_beta,
                                                             zero_division=0)})
    table = pd.DataFrame(rows)
    ok = table[table["flagged_share"] <= cfg.max_flag_share]
    pool = ok if not ok.empty else table
    best = pool.loc[pool[f"f{cfg.threshold_beta:g}"].idxmax()]
    return float(best["threshold"]), table


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float,
                      beta: float = settings.model.threshold_beta) -> Dict[str, object]:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(float(t), 3),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            f"f{beta:g}": round(fbeta_score(y, pred, beta=beta, zero_division=0), 4),
            "flagged_share": round(float(pred.mean()), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def tier_table(y: np.ndarray, p: np.ndarray, cutoffs: List[float],
               labels: Iterable[str] = settings.model.tier_labels) -> pd.DataFrame:
    """Share of crashes, observed High_Risk rate and lift per tier."""
    tiers = assign_tier(p, cutoffs)
    df = pd.DataFrame({"tier": tiers, "y": y})
    out = df.groupby("tier")["y"].agg(crashes="count", high_risk_rate="mean", severe="sum") \
        .reindex(list(labels)).fillna(0)
    out["share_of_crashes"] = out["crashes"] / len(df)
    out["share_of_severe"] = out["severe"] / max(df["y"].sum(), 1)
    out["lift"] = out["high_risk_rate"] / max(df["y"].mean(), 1e-9)
    return out.round(4)


# --------------------------------------------------------------------------- #
# Spatial (segment-level) risk tiers
# --------------------------------------------------------------------------- #
class SpatialRiskMapper:
    """
    Road-segment risk = crash frequency × shrunk severity propensity.

    ``expected_severe_per_year = crashes_per_year × p̂``, where p̂ is the
    segment's mean predicted probability shrunk toward the city mean with
    ``segment_shrinkage_k`` pseudo-crashes (empirical Bayes), so a segment
    with one unlucky crash is not labelled Very High. Tiers are quantiles of
    that quantity over segments with recorded crashes.
    """

    def __init__(self, cfg: ModelConfig = settings.model) -> None:
        self.cfg = cfg

    def fit(self, df: pd.DataFrame, p: np.ndarray, G: nx.MultiDiGraph
            ) -> Tuple[gpd.GeoDataFrame, Dict[Tuple[int, int], dict], List[float]]:
        ts = pd.to_datetime(df["timestamp"])
        years = max((ts.max() - ts.min()).days / 365.25, 1.0)
        key_u = np.minimum(df["edge_u"], df["edge_v"]).to_numpy()
        key_v = np.maximum(df["edge_u"], df["edge_v"]).to_numpy()
        seg = pd.DataFrame({"u": key_u, "v": key_v, "p": p}) \
            .groupby(["u", "v"])["p"].agg(crashes="count", p_sum="sum").reset_index()
        p_bar, k = float(np.mean(p)), self.cfg.segment_shrinkage_k
        seg["severity_propensity"] = (seg["p_sum"] + k * p_bar) / (seg["crashes"] + k)
        seg["crashes_per_year"] = seg["crashes"] / years
        seg["expected_severe_per_year"] = seg["crashes_per_year"] * seg["severity_propensity"]
        cutoffs = [float(q) for q in np.quantile(seg["expected_severe_per_year"],
                                                 self.cfg.tier_quantiles)]
        seg["tier"] = assign_tier(seg["expected_severe_per_year"].to_numpy(), cutoffs)

        e = edges_gdf(G).reset_index()
        e["u2"], e["v2"] = np.minimum(e["u"], e["v"]), np.maximum(e["u"], e["v"])
        e = e.drop_duplicates(["u2", "v2"])[["u2", "v2", "road_name", "road_class", "geometry"]]
        gdf = gpd.GeoDataFrame(seg.merge(e, left_on=["u", "v"], right_on=["u2", "v2"],
                                         how="left").drop(columns=["u2", "v2", "p_sum"]),
                               geometry="geometry", crs=e.crs)
        lookup = {(int(r.u), int(r.v)): {
            "tier": r.tier, "expected_severe_per_year": round(float(r.expected_severe_per_year), 4),
            "crashes_per_year": round(float(r.crashes_per_year), 3)}
            for r in seg.itertuples()}
        logger.info("Segment tiers over %d segments: %s", len(seg),
                    seg["tier"].value_counts().to_dict())
        return gdf, lookup, cutoffs


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class TrainingResult:
    metrics: pd.DataFrame
    chosen_model: str
    threshold: float
    tier_cutoffs: List[float]
    tiers_test: pd.DataFrame
    bundle_path: Optional[Path] = None
    details: Dict[str, dict] = field(default_factory=dict)
    bundle: Dict[str, object] = field(default_factory=dict, repr=False)


class TrainingPipeline:
    """End-to-end training run producing the API bundle and evaluation reports."""

    def __init__(self, G: nx.MultiDiGraph, feature_set: Optional[str] = None,
                 cfg: ModelConfig = settings.model, paths: PathConfig = settings.paths,
                 models: Iterable[str] = MODEL_NAMES,
                 oversampler: Optional[str] = None) -> None:
        self.G = RoadNetworkLoader().project(G)
        self.oversampler = oversampler or cfg.oversampler
        self.cfg, self.paths = cfg, paths
        self.feature_set = feature_set or cfg.feature_set
        self.models = tuple(models)
        self.splitter = SpatialSplitter(cfg)

    def run(self, features: gpd.GeoDataFrame, write: bool = True) -> TrainingResult:
        gdf = features.reset_index(drop=True)
        y_all = make_target(gdf).to_numpy()
        tr, te = self.splitter.train_test(gdf)
        train, test = gdf.iloc[tr].reset_index(drop=True), gdf.iloc[te].reset_index(drop=True)
        y_tr, y_te = y_all[tr], y_all[te]
        logger.info("Spatial split: train %d (%.1f%% high risk), test %d (%.1f%%)",
                    len(train), 100 * y_tr.mean(), len(test), 100 * y_te.mean())

        # Hotspots learned on the training blocks only
        with timed(logger, "DBSCAN on training split"):
            clus = HotspotClusteringPipeline(settings.clustering, self.paths) \
                .run(train, write=False)
        matcher = HotspotMatcher.from_result(clus)
        train = self._attach_hotspots(train, matcher.describe(clus.data["cluster_id"].to_numpy()))
        test = self._attach_hotspots(test, matcher.describe(
            matcher.match(np.column_stack([test.geometry.x, test.geometry.y]))))

        builder = ModelFeatureBuilder(self.cfg, self.feature_set).fit(train, pd.Series(y_tr))
        X_tr, X_te = builder.transform(train), builder.transform(test)
        groups = self.splitter.groups(train)

        rows, details, oof, fitted = [], {}, {}, {}
        for name in self.models:
            with timed(logger, f"{name}: out-of-fold + final fit"):
                pipe = build_pipeline(name, builder, float(y_tr.mean()), self.cfg,
                                      self.oversampler)
                oof[name] = cross_val_predict(pipe, X_tr, y_tr, groups=groups,
                                              cv=self.splitter.cv(), method="predict_proba")[:, 1]
                fitted[name] = pipe.fit(X_tr, y_tr)
                p_te = fitted[name].predict_proba(X_te)[:, 1]
            t_opt, t_table = optimise_threshold(y_tr, oof[name], self.cfg)
            at = {lbl: threshold_metrics(y_te, p_te, t) for lbl, t in
                  (("default_0.50", 0.50), (f"fixed_{self.cfg.fixed_threshold:.2f}",
                                            self.cfg.fixed_threshold), ("optimised", t_opt))}
            details[name] = {"test_proba": p_te, "thresholds": at, "threshold_table": t_table}
            rows.append({
                "model": name, "oof_pr_auc": average_precision_score(y_tr, oof[name]),
                "test_pr_auc": average_precision_score(y_te, p_te),
                "test_roc_auc": roc_auc_score(y_te, p_te),
                "test_brier": brier_score_loss(y_te, p_te), "optimised_threshold": t_opt,
                **{f"{k}_{m}": v[m] for k, v in at.items() for m in ("recall", "precision")},
            })
            logger.info("%s: test PR-AUC %.3f, ROC-AUC %.3f, threshold %.2f → recall %.3f",
                        name, rows[-1]["test_pr_auc"], rows[-1]["test_roc_auc"], t_opt,
                        at["optimised"]["recall"])

        metrics = pd.DataFrame(rows).round(4)
        metrics.insert(1, "prevalence_baseline", round(float(y_te.mean()), 4))
        # Selection on out-of-fold PR-AUC: the test set is never used to choose.
        best = metrics.sort_values("oof_pr_auc", ascending=False).iloc[0]["model"]
        threshold = float(metrics.set_index("model").loc[best, "optimised_threshold"])
        cutoffs = [float(q) for q in np.quantile(oof[best], self.cfg.tier_quantiles)]
        tiers_te = tier_table(y_te, details[best]["test_proba"], cutoffs)
        logger.info("Selected %s (threshold %.2f); tier cut points %s", best, threshold,
                    np.round(cutoffs, 3).tolist())

        # Spatial tiers from out-of-sample probabilities only (OOF on train, test preds)
        p_all = np.concatenate([oof[best], details[best]["test_proba"]])
        seg_gdf, seg_lookup, seg_cut = SpatialRiskMapper(self.cfg).fit(
            pd.concat([train, test], ignore_index=True), p_all, self.G)

        result = TrainingResult(metrics, best, threshold, cutoffs, tiers_te, details=details)
        result.bundle = {
            "model": fitted[best], "model_name": best, "builder": builder,
            "matcher": matcher, "threshold": threshold, "tier_cutoffs": cutoffs,
            "tier_labels": list(self.cfg.tier_labels), "segment_tiers": seg_lookup,
            "segment_tier_cutoffs": seg_cut, "feature_set": self.feature_set,
            "metrics": metrics.to_dict("records"),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_rows": len(train), "base_rate": float(y_tr.mean()),
            "graph_fingerprint": {"n_nodes": self.G.number_of_nodes(),
                                  "n_edges": self.G.number_of_edges(),
                                  "crs": str(self.G.graph.get("crs"))},
            "hotspot_params": clus.params,
        }
        if write:
            result.bundle_path = self._write(result, result.bundle, y_te, seg_gdf)
        return result

    @staticmethod
    def _attach_hotspots(df: gpd.GeoDataFrame, hot: pd.DataFrame) -> gpd.GeoDataFrame:
        df = df.drop(columns=[c for c in hot.columns if c in df]).reset_index(drop=True)
        return add_derived(pd.concat([df, hot.reset_index(drop=True)], axis=1))

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def _write(self, r: TrainingResult, bundle: dict, y_te: np.ndarray,
               seg_gdf: gpd.GeoDataFrame) -> Path:
        self.paths.ensure()
        suffix = "" if self.feature_set == "underwriting" else f"_{self.feature_set}"
        bundle_path = self.paths.model_bundle.with_name(f"risk_model{suffix}.joblib")
        joblib.dump(bundle, bundle_path, compress=3)

        rd = self.paths.reports_dir
        r.metrics.to_csv(rd / f"model_comparison{suffix}.csv", index=False)
        r.tiers_test.to_csv(rd / f"risk_tiers_test{suffix}.csv")
        json.dump({m: d["thresholds"] for m, d in r.details.items()},
                  open(rd / f"threshold_metrics{suffix}.json", "w"), indent=2)
        if not suffix:
            seg_gdf.to_parquet(self.paths.segment_risk, index=False)
        ReportPlotter(rd, suffix).all(r, y_te, seg_gdf if not suffix else None)
        logger.info("Saved bundle to %s and reports to %s", bundle_path, rd)
        return bundle_path


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
class ReportPlotter:
    """PR curves, confusion matrices, tier lift and segment-tier map (PNG)."""

    def __init__(self, out_dir: Path, suffix: str = "") -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt, self.out, self.suffix = plt, out_dir, suffix
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.edgecolor": "#8a8986",
                             "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                             "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb",
                             "axes.facecolor": "#fcfcfb"})

    def all(self, r: TrainingResult, y: np.ndarray, seg: Optional[gpd.GeoDataFrame]) -> None:
        self.pr_curves(r, y)
        self.confusion(r)
        self.tiers(r)
        if seg is not None:
            self.segment_map(seg)

    def _save(self, fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(self.out / f"{name}{self.suffix}.png", dpi=150)
        self.plt.close(fig)

    def pr_curves(self, r: TrainingResult, y: np.ndarray) -> None:
        fig, ax = self.plt.subplots(figsize=(6.4, 4.6))
        base = y.mean()
        ax.axhline(base, color="#8a8986", lw=1, ls="--")
        ax.text(0.99, base + 0.01, f"no-skill baseline ({base:.1%})", ha="right",
                va="bottom", color="#52514e", fontsize=8, transform=ax.get_yaxis_transform())
        m = r.metrics.set_index("model")
        for name, d in r.details.items():
            prec, rec, _ = precision_recall_curve(y, d["test_proba"])
            ax.plot(rec, prec, lw=2, color=MODEL_COLOURS[name],
                    label=f"{name.replace('_', ' ')} (PR-AUC {m.loc[name, 'test_pr_auc']:.3f})")
            t = d["thresholds"]["optimised"]
            ax.plot(t["recall"], t["precision"], "o", ms=8, color=MODEL_COLOURS[name],
                    mec="#fcfcfb", mew=2)
        ax.set_xlabel("Recall (share of severe/fatal crashes caught)")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color="#e8e7e3", lw=0.6)
        ax.legend(frameon=False, loc="upper right")
        ax.set_title("Precision–recall on held-out blocks (dots = chosen threshold)",
                     loc="left", color="#0b0b0b")
        self._save(fig, "pr_curves")

    def confusion(self, r: TrainingResult) -> None:
        keys = list(next(iter(r.details.values()))["thresholds"].keys())
        fig, axes = self.plt.subplots(len(r.details), len(keys),
                                      figsize=(2.4 * len(keys) + 0.6, 2.3 * len(r.details)))
        axes = np.atleast_2d(axes)
        for i, (name, d) in enumerate(r.details.items()):
            for j, k in enumerate(keys):
                c, ax = d["thresholds"][k]["confusion"], axes[i, j]
                mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
                share = mat / mat.sum(axis=1, keepdims=True)
                ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
                for (a, b), v in np.ndenumerate(mat):
                    ax.text(b, a, f"{v:,}\n{share[a, b]:.0%}", ha="center", va="center",
                            fontsize=8, color="#fcfcfb" if share[a, b] > 0.55 else "#0b0b0b")
                ax.set_xticks([0, 1], ["pred 0", "pred 1"])
                ax.set_yticks([0, 1], ["actual 0", "actual 1"])
                ax.set_title(f"{name.replace('_', ' ')}\n{k} (t={d['thresholds'][k]['threshold']:.2f})",
                             fontsize=8)
        self._save(fig, "confusion_matrices")

    def tiers(self, r: TrainingResult) -> None:
        t = r.tiers_test
        fig, ax = self.plt.subplots(figsize=(6.0, 3.6))
        bars = ax.bar(t.index, t["high_risk_rate"], color=TIER_COLOURS, width=0.6)
        for b, (_, row) in zip(bars, t.iterrows()):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{row['high_risk_rate']:.1%}\n{row['share_of_crashes']:.0%} of crashes",
                    ha="center", va="bottom", fontsize=8, color="#0b0b0b")
        ax.set_ylabel("Observed severe/fatal rate")
        ax.set_ylim(0, max(0.05, t["high_risk_rate"].max() * 1.35))
        ax.yaxis.set_major_formatter(self.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", color="#e8e7e3", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"4-tier risk on held-out blocks ({r.chosen_model.replace('_', ' ')})",
                     loc="left", color="#0b0b0b")
        self._save(fig, "risk_tier_lift")

    def segment_map(self, seg: gpd.GeoDataFrame) -> None:
        fig, ax = self.plt.subplots(figsize=(7.5, 8.5))
        seg = seg[seg.geometry.notna()]
        for tier, col, lw in zip(settings.model.tier_labels, TIER_COLOURS, (0.4, 0.7, 1.2, 2.0)):
            sub = seg[seg["tier"] == tier]
            if len(sub):
                sub.plot(ax=ax, color=col, linewidth=lw,
                         label=f"{tier} ({len(sub):,} segments)")
        ax.set_axis_off()
        ax.legend(frameon=False, loc="lower left")
        ax.set_title("Road-segment risk tiers — expected severe crashes per year",
                     loc="left", color="#0b0b0b")
        self._save(fig, "segment_risk_map")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Iterable[str]] = None) -> TrainingResult:
    parser = argparse.ArgumentParser(description="Train the collision severity-risk model.")
    parser.add_argument("-i", "--input", type=Path, default=settings.paths.features)
    parser.add_argument("--feature-set", choices=("underwriting", "claims_triage"),
                        default=settings.model.feature_set)
    parser.add_argument("--graphml", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    G = RoadNetworkLoader().load(args.graphml)
    res = TrainingPipeline(G, args.feature_set).run(gpd.read_parquet(args.input))
    print(res.metrics.to_string(index=False))
    print(res.tiers_test.to_string())
    return res


if __name__ == "__main__":
    main()
