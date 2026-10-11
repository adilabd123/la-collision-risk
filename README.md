# LA Vehicle Collision Insurance Risk Assessment System

Spatial risk engine built on 25,000 Los Angeles collision records.

## Layout
```
la-collision-risk/
├── config/
│   ├── __init__.py          # exports `settings`
│   └── config.py            # paths, CRS (EPSG:4326 → 32611), hyperparameters
├── src/
│   ├── utils.py             # logging / timing
│   ├── graph_utils.py       # shared OSMnx helpers (junctions, road classes)
│   ├── data_generator.py    # synthetic 25k-record LA dataset        (Step 1)
│   ├── ingestion.py         # validate → UTM reproject → OSMnx snap  (Step 1)
│   ├── features.py          # temporal + enrichment + collinearity   (Stage 2)
│   ├── clustering.py        # two-scale DBSCAN hotspots              (Stage 3)
│   ├── inference.py         # shared train/serve: hotspot matcher, scorer
│   └── train.py             # SMOTE, class weights, thresholds, tiers (Stage 4)
├── api/app.py               # FastAPI /predict service                (Stage 5)
├── models/                  # risk_model.joblib (served), _claims_triage variant
├── Dockerfile, requirements.txt, requirements-dev.txt
├── tests/                   # pytest, runs offline on a synthetic road grid
├── reports/                 # correlation matrix, VIF, DBSCAN grid, maps
└── data/{raw,interim,processed,cache}
```

## Quick start
```bash
pip install -r requirements.txt
python -m src.ingestion --force-download          # first run only: fetch + cache the LA network
python -m src.data_generator --graphml data/cache/la_drive_network.graphml
python -m src.ingestion                             # validate, reproject, snap
python -m src.features                              # features + collinearity reports
python -m src.clustering                            # DBSCAN hotspots
python -m src.train                                 # underwriting model → models/risk_model.joblib
python -m src.train --feature-set claims_triage     # optional post-crash model
uvicorn api.app:app --reload                        # http://localhost:8000/docs
pip install -r requirements-dev.txt && pytest -q    # 58 tests, offline
```
Passing `--graphml` to the generator plants intersection and corridor hotspots on the real
network (with `gt_hotspot_id` ground truth); without it, crashes are road-agnostic and DBSCAN
has no road structure to find. The first `ingestion` run downloads the City of LA drive network from OpenStreetMap
(several minutes, a few hundred MB); later runs load `data/cache/la_drive_network.graphml`.

Environment overrides use the `CRS_RISK_` prefix, e.g. `CRS_RISK_N_RECORDS=5000`,
`CRS_RISK_MAX_SNAP_M=30`.

## Outputs
* `data/processed/la_collisions_snapped.parquet` — GeoParquet, EPSG:32611; `geometry` is the
  snapped point, `raw_geometry` the original GPS point; adds `edge_u/v/key`, `snap_distance_m`,
  `snap_quality`, `road_class`, `speed_limit_mph`, `lanes`, `oneway`, `road_name`.
* `data/interim/la_collisions_rejected.csv` — every dropped row with a `reject_reason`.

* `data/processed/la_collisions_features.parquet` — all engineered features;
  `reports/selected_features.csv` lists the model-ready columns after screening.
* `data/processed/la_collisions_clustered.parquet` — per-crash `cluster_id`, `cluster_type`
  (`intersection` / `corridor` / `area` / `noise`), `hotspot_loss_rank`.
* `data/processed/la_hotspot_clusters.parquet` — one polygon per hotspot with shape, dominant
  road, local lift, severity and insured-loss totals.
* `reports/` — `feature_correlation_heatmap.png`, `feature_vif.csv`, `dbscan_tuning_grid.csv`,
  `k_distance_curve.png`, `hotspot_map.png`, `dbscan_groundtruth_eval.csv`.

Enrichment layers (`SimulatedWeatherProvider`, `SimulatedTrafficProvider`,
`SimulatedVenueSource`) implement the `EnrichmentProvider` / `VenueSource` interfaces; swap in a
real API client without touching the pipeline. `CRS_RISK_VENUE_SOURCE=osm` already switches POIs
to real OpenStreetMap bars/pubs/liquor stores.

Cluster membership is computed from all crashes: refit clustering inside each spatial CV fold
before using `in_hotspot` / `cluster_type` as model features, or it leaks.

`driver_gender` is kept for fairness auditing only and is listed in
`settings.model.protected_attributes` so Step 2 excludes it from rating features.

## Model (Stage 4)
* Target `High_Risk` = severe injury or fatal (≈11 %). Two feature sets:
  **underwriting** (served by default — only what is known before a crash) and
  **claims_triage** (adds collision type, speeding, alcohol, etc.).
* Spatial block split (2 km) for test and 5-fold CV; DBSCAN refit on training blocks only.
* SMOTE on the one-hot matrix, training folds only (inside an imblearn pipeline), to 1:2;
  `class_weight='balanced'` (RF) / `scale_pos_weight` (XGBoost) close the rest.
  SMOTENC is available (`CRS_RISK_OVERSAMPLER=smotenc`) but tested worse here.
* Decision threshold = max F2 on out-of-fold predictions, capped at flagging 35 % of crashes;
  0.50 and 0.20 reported alongside (`reports/threshold_metrics.json`).
* 4 tiers: crash level = out-of-fold probability quantiles (50/80/95 %); road-segment level =
  expected severe crashes per year (`data/processed/la_segment_risk_tiers.parquet`).

## API (Stage 5)
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "latitude": 34.0522, "longitude": -118.2437, "timestamp": "2025-11-14T23:30:00-08:00",
  "driver_age": 21, "vehicle_type": "motorcycle", "vehicle_year": 2009,
  "prior_claims_3yr": 1, "weather": "rain"}'
```
Returns probability, `high_risk` at the tuned threshold, `risk_tier`, snapped road, DBSCAN
hotspot match and the segment's spatial tier. Also `/predict/batch` (≤500, off-network rows
fail individually), `/model-info`, `/health`. `driver_gender` and unknown fields are rejected.

## Docker
```bash
docker build -t la-collision-risk .
docker run -p 8000:8000 -v "$(pwd)/data:/app/data" la-collision-risk
```
The model bundle is baked in; the road graph is mounted from `data/cache/`. The bundle records
which graph it was trained on and logs a warning if the served graph differs — **retrain after
downloading the real OSM network**, since hotspot and segment ids are graph-specific.
