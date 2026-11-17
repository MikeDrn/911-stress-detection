# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# One Dockerfile, three slim images (multi-stage targets):
#   --target api     -> stress-api     (FastAPI + PyTorch CPU + librosa)
#   --target ui      -> dispatcher-ui  (Streamlit only; no torch)
#   --target mlflow  -> mlflow-server  (MLflow tracking server + registry)
# Each target installs ONLY the uv extras it needs, so a UI change never
# rebuilds the 700 MB torch layer and the UI image stays small.
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false
RUN pip install --no-cache-dir "uv>=0.5,<1" \
 && groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
# Dependency manifests first: this layer is cached until they change.
# uv.loc[k] copies uv.lock when you have committed one, and is skipped otherwise.
COPY pyproject.toml README.md uv.loc[k] ./

# ----------------------------- stress-api ----------------------------------
FROM base AS api-deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --extra ml --extra api --extra tracking

FROM api-deps AS api
COPY params.yaml ./
COPY src ./src
RUN mkdir -p /app/telemetry /app/models && chown -R app:app /app/telemetry
USER app
EXPOSE 8000
# Readiness: 503 while weights load -> "starting"; 200 -> "healthy".
# start-period = 30 s startup budget during which failures are not counted.
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ---------------------------- dispatcher-ui --------------------------------
FROM base AS ui
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --extra ui
COPY src/ui.py ./src/ui.py
RUN mkdir -p /app/data/demo
USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).status == 200 else 1)"
CMD ["streamlit", "run", "src/ui.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

# ---------------------------- mlflow-server --------------------------------
FROM base AS mlflow
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --extra tracking
RUN mkdir -p /mlflow && chown -R app:app /mlflow
USER app
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4).status == 200 else 1)"
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", \
     "--host", "0.0.0.0", "--port", "5000", "--workers", "2", \
     "--allowed-hosts", "mlflow-server,mlflow-server:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*"]
