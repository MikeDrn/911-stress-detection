"""FastAPI inference service (``stress-api``).

Endpoints
---------
GET  /health/live      Liveness  - 200 while the process is up (never touches the model).
GET  /health           Readiness - 503 {"status": "loading"} while weights load,
                       503 {"status": "failed"} if loading failed, 200 {"status": "ready"}.
POST /predict          multipart WAV upload -> per-window duress scores + alert segments.
POST /admin/reload     Hot-swap to the current @champion without dropping traffic.
GET  /telemetry/calls  Recent audited calls (SQLite).
GET  /telemetry/calls/{call_id}
GET  /telemetry/stats

Model loading runs on a background thread started from the lifespan hook, so
the HTTP server answers probes immediately: Docker's healthcheck (and a
Kubernetes readiness probe, if this were ever moved to a cluster) sees 503
until the weights are in memory, and never routes a call to a half-loaded model.

Environment variables
---------------------
MODEL_SOURCE              auto | mlflow | file | fake       (default: auto)
MODEL_PATH                local checkpoint fallback        (default: models/champion.pt)
MLFLOW_TRACKING_URI       e.g. http://mlflow-server:5000
MODEL_NAME / MODEL_ALIAS  registry name / alias            (default: params.yaml / champion)
TELEMETRY_DB              SQLite path                      (default: telemetry/telemetry.db)
MODEL_LOAD_DELAY_SECONDS  artificial delay to demo the 503 -> 200 transition (default: 0)
FAKE_SCORES               comma-separated scores for MODEL_SOURCE=fake (demo/CI)
DURESS_THRESHOLD / CONSECUTIVE_WINDOWS   override params.yaml serving rule
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src import __version__
from src.alerting import find_alert_segments
from src.config import PROJECT_ROOT, load_params
from src.fake_model import FakeModel
from src.preprocessing import AudioContractError, AudioSpec, FeatureExtractor, preprocess_call
from src.telemetry import TelemetryStore

log = logging.getLogger("stress-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class Predictor(Protocol):
    version: str
    input_key: str

    def predict_proba(self, features: dict[str, np.ndarray]) -> np.ndarray: ...


LoaderResult = tuple[Predictor, str, str]  # (predictor, source, version)
Loader = Callable[[], LoaderResult]


# --------------------------------------------------------------------------- #
# Settings & model state
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    model_source: str = "auto"
    model_path: Path = PROJECT_ROOT / "models" / "champion.pt"
    mlflow_uri: str | None = None
    model_name: str = "voice-stress-detector"
    model_alias: str = "champion"
    telemetry_db: Path = PROJECT_ROOT / "telemetry" / "telemetry.db"
    load_delay_s: float = 0.0
    fake_scores: list[float] | None = None
    threshold: float = 0.82
    consecutive_windows: int = 3
    mlflow_retries: int = 5

    @classmethod
    def from_env(cls) -> Settings:
        p = load_params()
        fake = os.environ.get("FAKE_SCORES")
        return cls(
            model_source=os.environ.get("MODEL_SOURCE", "auto").lower(),
            model_path=Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models/champion.pt")),
            mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            model_name=os.environ.get("MODEL_NAME", p["mlflow"]["registered_model_name"]),
            model_alias=os.environ.get("MODEL_ALIAS", "champion"),
            telemetry_db=Path(
                os.environ.get("TELEMETRY_DB", PROJECT_ROOT / "telemetry" / "telemetry.db")
            ),
            load_delay_s=float(os.environ.get("MODEL_LOAD_DELAY_SECONDS", "0")),
            fake_scores=[float(s) for s in fake.split(",")] if fake else None,
            threshold=float(os.environ.get("DURESS_THRESHOLD", p["serving"]["duress_threshold"])),
            consecutive_windows=int(
                os.environ.get("CONSECUTIVE_WINDOWS", p["serving"]["consecutive_windows"])
            ),
            mlflow_retries=int(os.environ.get("MLFLOW_RETRIES", "5")),
        )


@dataclass
class ModelState:
    status: Literal["loading", "ready", "failed"] = "loading"
    predictor: Predictor | None = None
    source: str | None = None
    version: str | None = None
    error: str | None = None
    reloading: bool = False
    started_at: float = field(default_factory=time.time)
    ready_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def default_loader(settings: Settings) -> Loader:
    """Build the production loader: MLflow @champion -> local file (auto) | fake."""

    def _from_mlflow() -> LoaderResult:
        from src import registry
        from src.model import TorchStressPredictor

        if not settings.mlflow_uri:
            raise RuntimeError("MLFLOW_TRACKING_URI is not set")
        last: Exception | None = None
        for attempt in range(1, settings.mlflow_retries + 1):
            try:
                mv = registry.version_by_alias(
                    settings.mlflow_uri, settings.model_name, settings.model_alias
                )
                if mv is None:
                    raise LookupError(
                        f"no '{settings.model_name}@{settings.model_alias}' in registry"
                    )
                with tempfile.TemporaryDirectory() as td:
                    ckpt = registry.download_checkpoint(
                        settings.mlflow_uri, settings.model_name, settings.model_alias, td
                    )
                    version = f"{settings.model_name}@{settings.model_alias} v{mv.version}"
                    pred = TorchStressPredictor.from_checkpoint(ckpt, version=version)
                return pred, "mlflow", version
            except LookupError:
                raise
            except Exception as exc:  # server still booting, network blip ...
                last = exc
                log.warning("MLflow load attempt %d failed: %s", attempt, exc)
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError(f"MLflow unreachable after {settings.mlflow_retries} tries: {last}")

    def _from_file() -> LoaderResult:
        from src.model import TorchStressPredictor

        if not settings.model_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {settings.model_path}")
        mtime = datetime.fromtimestamp(settings.model_path.stat().st_mtime, UTC)
        version = f"file:{settings.model_path.name}@{mtime:%Y%m%dT%H%M%SZ}"
        return TorchStressPredictor.from_checkpoint(settings.model_path, version), "file", version

    def load() -> LoaderResult:
        if settings.load_delay_s > 0:
            log.info("simulating %.1fs weight-loading delay", settings.load_delay_s)
            time.sleep(settings.load_delay_s)
        src = settings.model_source
        if src == "fake":
            return FakeModel(settings.fake_scores, version="fake-demo"), "fake", "fake-demo"
        if src == "file":
            return _from_file()
        if src == "mlflow":
            return _from_mlflow()
        if src == "auto":
            try:
                return _from_mlflow()
            except Exception as exc:
                log.warning("MLflow champion unavailable (%s); falling back to local file", exc)
                return _from_file()
        raise ValueError(f"unknown MODEL_SOURCE '{src}'")

    return load


# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #
class WindowScore(BaseModel):
    index: int
    start_s: float
    end_s: float
    duress_score: float = Field(ge=0.0, le=1.0)
    above_threshold: bool
    in_alert: bool


class AlertSegmentOut(BaseModel):
    start_window: int
    end_window: int
    start_s: float
    end_s: float
    max_score: float


class PredictionResponse(BaseModel):
    call_id: str
    received_at: str
    filename: str | None
    duration_s: float
    n_windows: int
    threshold: float
    consecutive_windows: int
    alert: bool
    alert_segments: list[AlertSegmentOut]
    max_score: float
    mean_score: float
    windows: list[WindowScore]
    model_version: str
    model_source: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["loading", "ready", "failed"]
    model_source: str | None = None
    model_version: str | None = None
    error: str | None = None
    reloading: bool = False
    uptime_s: float
    threshold: float
    consecutive_windows: int
    service_version: str = __version__


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(loader: Loader | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    loader = loader or default_loader(settings)
    params = load_params()
    audio_spec = AudioSpec.from_params(params)
    extractor = FeatureExtractor(audio_spec)
    store = TelemetryStore(settings.telemetry_db)
    state = ModelState()

    def _warmup(predictor: Predictor) -> None:
        """Run one dummy window end-to-end so librosa/numba JIT and torch kernels are
        compiled *before* readiness flips to 200 (first real call stays fast)."""
        import io

        import soundfile as sf

        buf = io.BytesIO()
        noise = np.random.default_rng(0).normal(0, 0.01, audio_spec.window_samples)
        sf.write(buf, noise.astype(np.float32), audio_spec.sample_rate, format="WAV")
        predictor.predict_proba(preprocess_call(buf.getvalue(), audio_spec, extractor))

    def _load_blocking(initial: bool) -> None:
        try:
            predictor, source, version = loader()
            _warmup(predictor)
            with state.lock:
                state.predictor, state.source, state.version = predictor, source, version
                state.status, state.error, state.ready_at = "ready", None, time.time()
            log.info("model ready: source=%s version=%s", source, version)
        except Exception as exc:
            log.exception("model load failed")
            with state.lock:
                if initial or state.predictor is None:
                    state.status, state.error = "failed", f"{type(exc).__name__}: {exc}"
                else:  # failed hot-reload: keep serving the previous model
                    state.error = f"reload failed: {exc}"
        finally:
            state.reloading = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await store.init()
        app.state.load_task = asyncio.create_task(asyncio.to_thread(_load_blocking, True))
        yield

    app = FastAPI(
        title="911 Voice Stress Detection API",
        version=__version__,
        description="Decision-support scoring of acoustic stress in emergency calls. "
        "Scores are model confidences, not evidence of a threat.",
        lifespan=lifespan,
    )
    app.state.model = state
    app.state.settings = settings
    app.state.store = store

    def _health() -> HealthResponse:
        return HealthResponse(
            status=state.status,
            model_source=state.source,
            model_version=state.version,
            error=state.error,
            reloading=state.reloading,
            uptime_s=round(time.time() - state.started_at, 2),
            threshold=settings.threshold,
            consecutive_windows=settings.consecutive_windows,
        )

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "stress-api", "docs": "/docs", "health": "/health"}

    @app.get("/health/live")
    async def liveness():
        return {"status": "alive"}

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    async def readiness():
        body = _health()
        code = 200 if state.status == "ready" else 503
        return JSONResponse(status_code=code, content=body.model_dump())

    @app.post("/admin/reload", status_code=202)
    async def reload_model():
        if state.reloading:
            return {"status": "already reloading"}
        state.reloading = True
        app.state.reload_task = asyncio.create_task(asyncio.to_thread(_load_blocking, False))
        return {"status": "reloading", "current_version": state.version}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: Request,
        background: BackgroundTasks,
        file: UploadFile = File(..., description="16 kHz mono WAV"),
        call_id: str | None = Form(None),
    ):
        predictor = state.predictor
        if state.status != "ready" or predictor is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "model_not_ready", "status": state.status},
                headers={"Retry-After": "5"},
            )
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="file too large (max 50 MB)")

        t0 = time.perf_counter()
        try:
            feats = await run_in_threadpool(preprocess_call, raw, audio_spec, extractor)
        except AudioContractError as exc:
            raise HTTPException(
                422, detail={"error": "audio_contract_violation", "message": str(exc)}
            ) from exc
        scores = await run_in_threadpool(predictor.predict_proba, feats)
        scores = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        latency_ms = (time.perf_counter() - t0) * 1000

        starts, ends = feats["starts"], feats["ends"]
        segments = find_alert_segments(
            scores, starts, ends, settings.threshold, settings.consecutive_windows
        )
        in_alert = np.zeros(len(scores), dtype=bool)
        for seg in segments:
            in_alert[seg.start_window : seg.end_window + 1] = True

        windows = [
            WindowScore(
                index=i,
                start_s=round(float(starts[i]), 3),
                end_s=round(float(ends[i]), 3),
                duress_score=round(float(s), 5),
                above_threshold=bool(s > settings.threshold),
                in_alert=bool(in_alert[i]),
            )
            for i, s in enumerate(scores)
        ]
        resp = PredictionResponse(
            call_id=call_id or f"call-{uuid.uuid4().hex[:12]}",
            received_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
            filename=file.filename,
            duration_s=round(float(feats["duration_s"]), 3),
            n_windows=len(windows),
            threshold=settings.threshold,
            consecutive_windows=settings.consecutive_windows,
            alert=bool(segments),
            alert_segments=[
                AlertSegmentOut(**{**s.to_dict(), "max_score": round(s.max_score, 5)})
                for s in segments
            ],
            max_score=round(float(scores.max()), 5),
            mean_score=round(float(scores.mean()), 5),
            windows=windows,
            model_version=str(predictor.version),
            model_source=str(state.source),
            latency_ms=round(latency_ms, 2),
        )
        background.add_task(_audit, store, resp, request.client.host if request.client else None)
        return resp

    @app.get("/telemetry/calls")
    async def recent_calls(limit: int = 50):
        return await store.recent_calls(min(max(limit, 1), 500))

    @app.get("/telemetry/calls/{call_id}")
    async def call_detail(call_id: str):
        call = await store.get_call(call_id)
        if call is None:
            raise HTTPException(404, detail="unknown call_id")
        return call

    @app.get("/telemetry/stats")
    async def telemetry_stats():
        return await store.stats()

    return app


async def _audit(store: TelemetryStore, resp: PredictionResponse, client: str | None) -> None:
    """Background task: never let an audit-log failure surface to the caller."""
    try:
        call: dict[str, Any] = resp.model_dump(exclude={"windows", "alert_segments", "mean_score"})
        call.update(n_alert_segments=len(resp.alert_segments), client=client)
        await store.log_call(call, [w.model_dump() for w in resp.windows])
    except Exception:
        log.exception("telemetry write failed for %s", resp.call_id)


app = create_app()
