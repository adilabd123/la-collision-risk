# LA Collision Insurance Risk API
# Build:  docker build -t la-collision-risk .
# Run:    docker run -p 8000:8000 -v "$(pwd)/data:/app/data" la-collision-risk
#
# The trained bundle (models/) is baked into the image. The road graph is NOT:
# it is large and region-specific, so mount data/ (containing
# data/cache/la_drive_network.graphml) or point CRS_RISK_GRAPHML elsewhere.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# libgomp: OpenMP runtime needed by XGBoost and scikit-learn wheels.
# GEOS/PROJ/GDAL come bundled inside the shapely/pyproj/pyogrio wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code and trained model bundle
COPY config/ config/
COPY src/ src/
COPY api/ api/
COPY models/ models/

RUN useradd --create-home --uid 10001 app \
 && mkdir -p data/cache data/processed logs reports \
 && chown -R app:app /app
USER app

ENV CRS_RISK_PROJECT_ROOT=/app \
    CRS_RISK_MODEL_PATH=/app/models/risk_model.joblib \
    CRS_RISK_GRAPHML=/app/data/cache/la_drive_network.graphml \
    CRS_RISK_LOG_LEVEL=INFO

EXPOSE 8000

# start-period covers loading a city-scale road graph (~30-60 s for LA).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# One worker: each worker holds its own copy of the road graph in memory.
# Scale with more containers rather than more workers.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
