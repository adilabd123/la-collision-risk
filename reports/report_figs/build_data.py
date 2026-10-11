"""Collect every number used in the report from project outputs into data.json."""
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/la-collision-risk")
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, joblib, osmnx as ox
from sklearn.metrics import roc_auc_score, average_precision_score
from config import settings
from src.train import make_target, SpatialSplitter
from src.clustering import KDistanceAnalyzer

ROOT = Path("/home/claude/la-collision-risk"); REP = ROOT / "reports"
SP = Path("/tmp/claude-0/-home-claude/a45bdf43-03f5-5920-a059-b11e1cd584c4/scratchpad")
D = {}

raw = pd.read_csv(settings.paths.raw_collisions)
rej = pd.read_csv(settings.paths.rejected_records)
f = gpd.read_parquet(settings.paths.features).reset_index(drop=True)
y = make_target(f).to_numpy()
D.update(raw_rows=len(raw), raw_cols=raw.shape[1], clean_rows=len(f), high_risk_rate=float(y.mean()),
         rejected=rej["reject_reason"].value_counts().to_dict(), rejected_total=len(rej),
         snap_median=float(f.snap_distance_m.median()), snap_p95=float(f.snap_distance_m.quantile(.95)),
         max_flag_share=settings.model.max_flag_share)
G = ox.load_graphml(SP / "grid_city.graphml")
D.update(grid_nodes=G.number_of_nodes(), grid_edges=G.number_of_edges())
D["vif_max"] = float(pd.read_csv(REP / "feature_vif.csv", index_col=0).iloc[:, 0].max())
D["corr_pairs"] = len(pd.read_csv(REP / "collinear_pairs.csv"))
tg = pd.read_csv(REP / "dbscan_tuning_grid.csv")
D.update(grid_int=int((tg.stage == "intersection").sum()), grid_cor=int((tg.stage == "corridor").sum()))
tr, te = SpatialSplitter().train_test(f)
D.update(train_rows=len(tr), test_rows=len(te))

# Severity
labels = {"property_damage_only": "Property damage only", "minor_injury": "Minor injury",
          "severe_injury": "Severe injury", "fatal": "Fatal"}
sev = []
for k, lab in labels.items():
    m = f.severity == k
    sev.append({"label": lab, "count": int(m.sum()), "share": float(m.mean()),
                "median_claim": float(f.loc[m, "claim_amount_usd"].median())})
D["severity"] = sev
D["claim_ratio_fatal_pdo"] = int(round(sev[3]["median_claim"] / sev[0]["median_claim"]))
D["driver_age_range"] = [int(f.driver_age.min()), int(f.driver_age.max())]

# High-risk rate by condition
f["HR"] = y
ts = pd.to_datetime(f.timestamp)
night = ts.dt.hour.isin(settings.features.night_hours)
rush = ts.dt.hour.isin(settings.features.rush_hours) & (ts.dt.dayofweek < 5)
rate = lambda m: float(f.loc[m, "HR"].mean())
vt = f.vehicle_type.astype(str); lt = f.lighting.astype(str); wx = f.weather.astype(str)
age = f.driver_age
cond = [("Vehicle type", "Motorcycle", vt == "motorcycle"), ("Vehicle type", "Bicycle", vt == "bicycle"),
        ("Vehicle type", "Commercial truck", vt == "commercial_truck"), ("Vehicle type", "Sedan", vt == "sedan"),
        ("Lighting", "Dark, no street lights", lt == "dark_no_street_lights"),
        ("Lighting", "Dark, street lights", lt == "dark_street_lights"), ("Lighting", "Daylight", lt == "daylight"),
        ("Time", "Night (21:00–04:59)", night), ("Time", "Day", ~night),
        ("Time", "Weekday rush hour", rush), ("Time", "Other times", ~rush),
        ("Weather", "Rain", wx == "rain"), ("Weather", "Clear", wx == "clear"),
        ("Driver age", "16–24", age <= 24), ("Driver age", "25–64", (age >= 25) & (age <= 64)),
        ("Driver age", "65 and over", age >= 65), ("All collisions", "–", pd.Series(True, index=f.index))]
D["conditions"] = [{"cond": c, "group": g, "rate": rate(m)} for c, g, m in cond]
D["cond_lookup"] = {"motorcycle": rate(vt == "motorcycle"), "bicycle": rate(vt == "bicycle"),
                    "sedan": rate(vt == "sedan"), "dark_no_lights": rate(lt == "dark_no_street_lights"),
                    "young": rate(age <= 24), "rush": rate(rush), "not_rush": rate(~rush)}

# DBSCAN
D["knee"] = float(KDistanceAnalyzer(settings.clustering.knn_k).fit(np.c_[f.geometry.x, f.geometry.y]).knee_eps_m)
ev = pd.read_csv(REP / "dbscan_groundtruth_eval.csv", index_col=0)["value"].to_dict()
sel = {}
for st, key in (("intersection", "int"), ("corridor", "cor")):
    row = tg[(tg.stage == st)].sort_values(["feasible", "capture_score"], ascending=False).iloc[0]
    sel[key] = {"eps": int(row.eps_m), "ms": int(row.min_samples), "n": int(row.n_clusters),
                "coverage": float(row.coverage), "pai": float(row.pai), "area_km2": float(row.hotspot_area_km2)}
cl = gpd.read_parquet(settings.paths.cluster_summary)
types = cl.cluster_type.value_counts().to_dict()
cf = gpd.read_parquet(settings.paths.clustered)
in_hot = cf.cluster_id >= 0
D["db"] = {**sel, "eval": ev, "types": types, "total": len(cl),
           "int_recovered": int(round(ev["intersection_recall"] * ev["intersection_detectable"])),
           "cor_recovered": int(round(ev["corridor_recall"] * ev["corridor_detectable"])),
           "coverage_total": float(in_hot.mean()), "area_km2": sel["int"]["area_km2"] + sel["cor"]["area_km2"],
           "claims_share": float(cf.loc[in_hot, "claim_amount_usd"].sum() / cf.claim_amount_usd.sum()),
           "severe_share_in_hotspots": float((cf.loc[in_hot, "severity_code"] >= 2).mean()),
           "severe_share_outside": float((cf.loc[~in_hot, "severity_code"] >= 2).mean())}
D["top_hotspots"] = cl.sort_values("loss_rank").head(8)[["loss_rank", "cluster_type", "n_collisions", "length_m",
    "dominant_road", "severe_fatal_share", "total_claims_usd"]].to_dict("records")

# Models
mc = pd.read_csv(REP / "model_comparison.csv")
nice = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
D["models"] = [{"name": nice[r.model], "oof_pr_auc": r.oof_pr_auc, "test_pr_auc": r.test_pr_auc,
                "test_roc_auc": r.test_roc_auc, "test_brier": r.test_brier, "threshold": r.optimised_threshold,
                "recall": r.optimised_recall, "precision": r.optimised_precision} for r in mc.itertuples()]
D["base_test"] = float(mc.prevalence_baseline.iloc[0])
osf = pd.read_csv(SP / "os_underwriting_final.csv")
D["os"] = {v: {"lr": float(osf[(osf.oversampler == v) & (osf.model == "logistic_regression")].test_pr_auc.iloc[0]),
               "rf": float(osf[(osf.oversampler == v) & (osf.model == "random_forest")].test_pr_auc.iloc[0]),
               "xgb": float(osf[(osf.oversampler == v) & (osf.model == "xgboost")].test_pr_auc.iloc[0])}
           for v in ("smote", "smotenc", "none")}
thm = json.load(open(REP / "threshold_metrics.json"))["logistic_regression"]
D["thresholds"] = [{"label": lab, "recall": thm[k]["recall"], "precision": thm[k]["precision"], "f2": thm[k]["f2"],
                    "flagged": thm[k]["flagged_share"]} for k, lab in
                   (("default_0.50", "0.50 (default)"), ("fixed_0.20", "0.20 (fixed)"),
                    ("optimised", f"{thm['optimised']['threshold']:.2f} (tuned, F2 with 35% cap)"))]
D["thr"] = thm["optimised"]["threshold"]; D["conf"] = thm["optimised"]["confusion"]

# Feature sets + oracle ceilings (generator's own severity formula, before noise)
mt = pd.read_csv(REP / "model_comparison_claims_triage.csv")
f_ = lambda m: np.asarray(m, dtype=float)
b = lambda c: f_(f[c].astype(str).isin(["True", "1", "true"]))
veh_age = (ts.dt.year - f.vehicle_year).to_numpy(float); a = age.to_numpy(float)
ex_ante = (1.4 * f_(vt.isin(["motorcycle", "bicycle"])) + 0.5 * f_(vt == "commercial_truck")
           + 0.25 * f_(lt == "dark_street_lights") + 0.55 * f_(lt == "dark_no_street_lights")
           + 0.30 * f_(f.road_surface.astype(str) == "wet") + 0.35 * f_(a < 25)
           + 0.025 * np.clip(a - 65, 0, None) + 0.015 * veh_age)
ct = f.collision_type.astype(str)
full = ex_ante + 0.9 * b("speeding") + 1.2 * b("alcohol_involved") + 0.3 * b("distracted") \
    + 0.9 * f_(ct == "pedestrian") + 0.7 * f_(ct == "head_on")
D["feature_sets"] = [
    {"name": "Underwriting (pre-crash)", "pr": float(mc.test_pr_auc.max()), "roc": float(mc.test_roc_auc.max()),
     "ceil_pr": float(average_precision_score(y[te], ex_ante[te])), "ceil_roc": float(roc_auc_score(y[te], ex_ante[te]))},
    {"name": "Claims triage (+ post-crash)", "pr": float(mt.test_pr_auc.max()), "roc": float(mt.test_roc_auc.max()),
     "ceil_pr": float(average_precision_score(y[te], full[te])), "ceil_roc": float(roc_auc_score(y[te], full[te]))}]

# Importance, tiers, segments, areas
imp = pd.read_csv(REP / "report_figs/feature_importance.csv")
D["importance"] = imp.head(10)[["label", "mean", "std"]].to_dict("records")
D["importance_hotspot_rank"] = {r.feature: i + 1 for i, r in enumerate(imp.itertuples())
                                if r.feature in ("cluster_id", "in_hotspot", "hotspot_type")}
D["tiers"] = pd.read_csv(REP / "risk_tiers_test.csv").to_dict("records")
seg = gpd.read_parquet(settings.paths.segment_risk)
D["segments"] = {**seg.tier.value_counts().to_dict(), "total": len(seg)}
ar = pd.read_csv(REP / "report_figs/area_ranking.csv").head(10)
D["areas"] = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
              for r in ar.to_dict("records")]
D["area_rate_min"], D["area_rate_max"] = float(ar.high_risk_rate.min()), float(ar.high_risk_rate.max())
D["latency_ms"] = 97

D["images"] = json.load(open(SP / "report/images.json"))
D["api_request"] = json.load(open(SP / "report/api_example.json"))["request"]
D["api_response"] = json.load(open(SP / "report/api_example.json"))["response"]

import importlib.metadata as md
libs = [("pandas", "Tabular data handling"), ("numpy", "Numerical computing"),
        ("geopandas", "Spatial data, reprojection, GeoParquet"), ("shapely", "Geometry, spatial indexes, linear referencing"),
        ("pyproj", "Coordinate transformations"), ("osmnx", "Road network download and graph handling"),
        ("networkx", "Graph data structures"), ("scikit-learn", "Models, DBSCAN, metrics, target encoding"),
        ("imbalanced-learn", "SMOTE / SMOTENC inside pipelines"), ("xgboost", "Gradient-boosted trees"),
        ("matplotlib", "Figures"), ("fastapi", "Web API"), ("uvicorn", "API server"), ("pytest", "Automated tests")]
D["libraries"] = [[n, md.version(n), p] for n, p in libs]

D["abbreviations"] = [["API", "Application Programming Interface"], ["AUC", "Area Under the Curve"],
    ["CRS", "Coordinate Reference System"], ["CV", "Cross-Validation"],
    ["DBSCAN", "Density-Based Spatial Clustering of Applications with Noise"], ["EPSG", "European Petroleum Survey Group (CRS registry)"],
    ["F2", "F-score weighting recall twice as much as precision (β = 2)"], ["GIS", "Geographic Information System"],
    ["LA", "Los Angeles"], ["LAPD", "Los Angeles Police Department"], ["LR", "Logistic Regression"],
    ["ML", "Machine Learning"], ["OOF", "Out-of-Fold"], ["OSM", "OpenStreetMap"], ["PAI", "Prediction Accuracy Index"],
    ["PCA", "Principal Component Analysis"], ["PR-AUC", "Area Under the Precision–Recall Curve"],
    ["RF", "Random Forest"], ["ROC", "Receiver Operating Characteristic"],
    ["SMOTE", "Synthetic Minority Over-sampling Technique"], ["SMOTENC", "SMOTE for Nominal and Continuous features"],
    ["UTM", "Universal Transverse Mercator"], ["VIF", "Variance Inflation Factor"], ["XGB", "Extreme Gradient Boosting (XGBoost)"]]

D["references"] = [
    "Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. {i}Computers, Environment and Urban Systems, 65{/i}, 126–139.",
    "Breiman, L. (2001). Random forests. {i}Machine Learning, 45{/i}(1), 5–32.",
    "Chainey, S., Tompson, L., & Uhlig, S. (2008). The utility of hotspot mapping for predicting spatial patterns of crime. {i}Security Journal, 21{/i}(1–2), 4–28.",
    "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. {i}Journal of Artificial Intelligence Research, 16{/i}, 321–357.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In {i}Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining{/i} (pp. 785–794).",
    "Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In {i}Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96){/i} (pp. 226–231).",
    "Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. {i}Journal of Machine Learning Research, 18{/i}(17), 1–5.",
    "Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. {i}ACM SIGKDD Explorations Newsletter, 3{/i}(1), 27–32.",
    "Okabe, A., & Sugihara, K. (2012). {i}Spatial analysis along networks: Statistical and computational methods{/i}. Wiley.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. {i}Journal of Machine Learning Research, 12{/i}, 2825–2830.",
    "Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. {i}Ecography, 40{/i}(8), 913–929.",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. {i}PLoS ONE, 10{/i}(3), e0118432.",
]

sev_r = D["high_risk_rate"]; db = D["db"]; tiers = D["tiers"]
pr_lo = min(m["test_pr_auc"] for m in D["models"]); pr_hi = max(m["test_pr_auc"] for m in D["models"])
D["abstract"] = (
    "Insurers need to know not only how often collisions happen but how severe they are likely to be and where severe outcomes concentrate. "
    f"This project develops a geospatial risk assessment system for vehicle collision insurance in Los Angeles using {D['raw_rows']:,} collision records from 2021 to 2025. "
    f"After validation, {D['clean_rows']:,} records were reprojected from WGS 84 to UTM Zone 11N (EPSG:32611) and snapped to the nearest road segment. "
    "Each collision was enriched with temporal, weather, road-friction, traffic, venue and intersection features, and redundant features were removed through Pearson-correlation and VIF screening. "
    f"A two-scale DBSCAN procedure, tuned with the Prediction Accuracy Index, identified {db['total']} hotspots ({db['types']['intersection']} intersections, {db['types']['corridor']} corridors and {db['types']['area']} areas) that contain {100*db['coverage_total']:.1f}% of collisions on about {db['area_km2']:.1f} km² of road footprint and account for {100*db['claims_share']:.0f}% of insured losses. "
    f"Severe or fatal collisions ({100*sev_r:.1f}%) were predicted with Logistic Regression, Random Forest and XGBoost, using SMOTE on training folds only, cost-sensitive class weights and spatial block cross-validation. "
    f"On held-out city blocks the models reached PR-AUC {pr_lo:.2f}–{pr_hi:.2f} against a baseline of {D['base_test']:.2f} using only information available before a crash, close to the best achievable value of {D['feature_sets'][0]['ceil_pr']:.2f}; adding post-crash details raised PR-AUC to {D['feature_sets'][1]['pr']:.2f}. "
    f"A tuned decision threshold of {D['thr']:.2f} caught {100*D['thresholds'][2]['recall']:.0f}% of severe collisions while flagging a third of cases, whereas a fixed threshold of 0.20 flagged {100*D['thresholds'][1]['flagged']:.0f}%. "
    f"Predictions were converted into a four-tier risk model whose Very High tier had a severe-collision rate of {100*tiers[3]['high_risk_rate']:.1f}%, {tiers[3]['lift']:.1f} times the base rate, and into road-segment risk zones. "
    "The model is served through a FastAPI web service packaged for Docker. The data are synthetic and the road network was a synthetic grid, so the findings demonstrate the method and must be confirmed on real records.")

json.dump(D, open(SP / "report/data.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: D[k] for k in ("claim_ratio_fatal_pdo", "driver_age_range", "knee", "corr_pairs", "vif_max", "grid_int", "grid_cor",
                                    "importance_hotspot_rank", "os", "feature_sets", "area_rate_min", "area_rate_max")}, indent=1, default=str))
print("hotspot severe share", db["severe_share_in_hotspots"], "outside", db["severe_share_outside"], "coverage", db["coverage_total"], "area", db["area_km2"], "claims", db["claims_share"])
print("cond", D["cond_lookup"])
