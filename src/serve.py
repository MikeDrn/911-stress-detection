"""FastAPI inference service (``stress-api``).

Endpoints
---------
GET  /health/live        Liveness  - 200 while the process is up.
GET  /health             Readiness - 503 while the champion loads / if loading failed, else 200.
POST /predict            multipart WAV -> per-window duress scores + alert segments.
POST /feedback           dispatcher ground truth for a call (delayed label, Course 4).
GET  /rollout            current release experiment (mode, canary step, split ...).
POST /rollout            start an experiment: shadow | canary | ab | bandit | champion_only.
GET  /rollout/report     live statistics + decision for the current experiment.
GET  /rollout/events     audit trail of release decisions.
POST /admin/reload       re-load @champion and @challenger without downtime.
GET  /telemetry/...      audited calls and stats.

Release modes (Course 4)
------------------------
champion_only  every call is scored by @champion.
shadow         @champion answers; @challenger scores the same call *after* the
               response is sent (mirrored traffic, never shown to the dispatcher).
canary         a sticky slice (10% -> 25% -> 50% -> 100%) is served by
               @challenger; guardrails are evaluated automatically after every
               call and the canary advances or rolls back on its own.
ab             fixed 50/50 sticky split with a pre-computed sample size.
bandit         adaptive split (Thompson / UCB1 / epsilon-greedy, 5% floor);
               the propensity of every routing decision is logged.

Environment variables
---------------------
MODEL_SOURCE              auto | mlflow | file | fake       (default: auto)
MODEL_PATH                local champion checkpoint fallback
CHALLENGER_PATH           local challenger checkpoint (MODEL_SOURCE=file)
MLFLOW_TRACKING_URI       e.g. http://mlflow-server:5000
MODEL_NAME                registry name (default: params.yaml)
TELEMETRY_DB              SQLite path
MODEL_LOAD_DELAY_SECONDS  artificial delay to demo the 503 -> 200 transition
MODEL_POLL_SECONDS        how often to check the registry aliases (0 = never, default 30)
FAKE_SCORES / FAKE_CHALLENGER_SCORES   test doubles for MODEL_SOURCE=fake
DURESS_THRESHOLD / CONSECUTIVE_WINDOWS override params.yaml serving rule
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

from src import __version__, bandits
from src import rollout as ro
from src.alerting import find_alert_segments
from src.config import PROJECT_ROOT, load_params
from src.fake_model import FakeModel
from src.preprocessing import AudioContractError, AudioSpec, FeatureExtractor, preprocess_call
from src.telemetry import TelemetryStore, now_iso

log = logging.getLogger("stress-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class Predictor(Protocol):
    version: str
    input_key: str

    def predict_proba(self, features: dict[str, np.ndarray]) -> np.ndarray: ...


LoaderResult = tuple[Predictor, str, str]  # (predictor, source, version)
Loader = Callable[[], LoaderResult]
ChallengerLoader = Callable[[], LoaderResult | None]


# --------------------------------------------------------------------------- #
# Settings & model state
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    model_source: str = "auto"
    model_path: Path = PROJECT_ROOT / "models" / "champion.pt"
    challenger_path: Path = PROJECT_ROOT / "models" / "challenger.pt"
    mlflow_uri: str | None = None
    model_name: str = "voice-stress-detector"
    telemetry_db: Path = PROJECT_ROOT / "telemetry" / "telemetry.db"
    load_delay_s: float = 0.0
    fake_scores: list[float] | None = None
    fake_challenger_scores: list[float] | None = None
    threshold: float = 0.82
    consecutive_windows: int = 3
    mlflow_retries: int = 5
    poll_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> Settings:
        p = load_params()

        def floats(name: str) -> list[float] | None:
            raw = os.environ.get(name)
            return [float(s) for s in raw.split(",")] if raw else None

        return cls(
            model_source=os.environ.get("MODEL_SOURCE", "auto").lower(),
            model_path=Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models/champion.pt")),
            challenger_path=Path(
                os.environ.get("CHALLENGER_PATH", PROJECT_ROOT / "models/challenger.pt")
            ),
            mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            model_name=os.environ.get("MODEL_NAME", p["mlflow"]["registered_model_name"]),
            telemetry_db=Path(
                os.environ.get("TELEMETRY_DB", PROJECT_ROOT / "telemetry" / "telemetry.db")
            ),
            load_delay_s=float(os.environ.get("MODEL_LOAD_DELAY_SECONDS", "0")),
            fake_scores=floats("FAKE_SCORES"),
            fake_challenger_scores=floats("FAKE_CHALLENGER_SCORES"),
            threshold=float(os.environ.get("DURESS_THRESHOLD", p["serving"]["duress_threshold"])),
            consecutive_windows=int(
                os.environ.get("CONSECUTIVE_WINDOWS", p["serving"]["consecutive_windows"])
            ),
            mlflow_retries=int(os.environ.get("MLFLOW_RETRIES", "5")),
            poll_seconds=float(os.environ.get("MODEL_POLL_SECONDS", "30")),
        )


@dataclass
class ModelState:
    status: Literal["loading", "ready", "failed"] = "loading"
    predictor: Predictor | None = None
    source: str | None = None
    version: str | None = None
    challenger: Predictor | None = None
    challenger_version: str | None = None
    error: str | None = None
    reloading: bool = False
    started_at: float = field(default_factory=time.time)
    ready_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _registry_version(settings: Settings, alias: str) -> str | None:
    from src import registry

    mv = registry.version_by_alias(settings.mlflow_uri, settings.model_name, alias)
    return None if mv is None else f"{settings.model_name}@{alias} v{mv.version}"


def _load_alias(settings: Settings, alias: str) -> LoaderResult:
    from src import registry
    from src.model import TorchStressPredictor

    if not settings.mlflow_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")
    last: Exception | None = None
    for attempt in range(1, settings.mlflow_retries + 1):
        try:
            version = _registry_version(settings, alias)
            if version is None:
                raise LookupError(f"no '{settings.model_name}@{alias}' in registry")
            with tempfile.TemporaryDirectory() as td:
                ckpt = registry.download_checkpoint(
                    settings.mlflow_uri, settings.model_name, alias, td
                )
                pred = TorchStressPredictor.from_checkpoint(ckpt, version=version)
            return pred, "mlflow", version
        except LookupError:
            raise
        except Exception as exc:  # server still booting, network blip ...
            last = exc
            log.warning("MLflow load of @%s attempt %d failed: %s", alias, attempt, exc)
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"MLflow unreachable after {settings.mlflow_retries} tries: {last}")


def _load_file(path: Path, label: str) -> LoaderResult:
    from src.model import TorchStressPredictor

    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    version = f"file:{path.name}@{mtime:%Y%m%dT%H%M%SZ}"
    return TorchStressPredictor.from_checkpoint(path, version), "file", version


def default_loader(settings: Settings) -> Loader:
    """Champion loader: MLflow @champion -> local file (auto) | file | fake."""

    def load() -> LoaderResult:
        if settings.load_delay_s > 0:
            log.info("simulating %.1fs weight-loading delay", settings.load_delay_s)
            time.sleep(settings.load_delay_s)
        src = settings.model_source
        if src == "fake":
            return FakeModel(settings.fake_scores, version="fake-champion"), "fake", "fake-champion"
        if src == "file":
            return _load_file(settings.model_path, "champion")
        if src == "mlflow":
            return _load_alias(settings, "champion")
        if src == "auto":
            try:
                return _load_alias(settings, "champion")
            except Exception as exc:
                log.warning("MLflow champion unavailable (%s); falling back to local file", exc)
                return _load_file(settings.model_path, "champion")
        raise ValueError(f"unknown MODEL_SOURCE '{src}'")

    return load


def default_challenger_loader(settings: Settings) -> ChallengerLoader:
    """Challenger loader. Returns None when there is nothing to compare."""

    def load() -> LoaderResult | None:
        src = settings.model_source
        if src == "fake":
            if settings.fake_challenger_scores is None:
                return None
            fake = FakeModel(settings.fake_challenger_scores, version="fake-challenger")
            return fake, "fake", "fake-challenger"
        if src == "file":
            return (
                _load_file(settings.challenger_path, "challenger")
                if (settings.challenger_path.exists())
                else None
            )
        try:
            champ = _registry_version(settings, "champion")
            chall = _registry_version(settings, "challenger")
        except Exception as exc:
            log.warning("cannot read @challenger alias: %s", exc)
            return None
        if chall is None or (champ and champ.split(" v")[-1] == chall.split(" v")[-1]):
            return None  # challenger is the champion itself: nothing to evaluate
        return _load_alias(settings, "challenger")

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
    served_by: str = "champion"
    rollout_mode: str = "champion_only"


class HealthResponse(BaseModel):
    status: Literal["loading", "ready", "failed"]
    model_source: str | None = None
    model_version: str | None = None
    challenger_version: str | None = None
    rollout_mode: str | None = None
    error: str | None = None
    reloading: bool = False
    uptime_s: float
    threshold: float
    consecutive_windows: int
    service_version: str = __version__


class FeedbackIn(BaseModel):
    call_id: str
    true_label: Literal["stressed", "calm"]


class RolloutIn(BaseModel):
    mode: Literal["champion_only", "shadow", "canary", "ab", "bandit"]
    policy: Literal["thompson", "ucb1", "epsilon_greedy"] | None = None
    baseline_rate: float | None = Field(None, gt=0, lt=1)
    mde: float | None = Field(None, gt=0, lt=1)


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(
    loader: Loader | None = None,
    settings: Settings | None = None,
    challenger_loader: ChallengerLoader | None = None,
    params: dict[str, Any] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    injected = loader is not None
    loader = loader or default_loader(settings)
    if challenger_loader is None:
        challenger_loader = (lambda: None) if injected else default_challenger_loader(settings)
    params = params or load_params()
    rcfg = params.get("rollout", {})
    audio_spec = AudioSpec.from_params(params)
    extractor = FeatureExtractor(audio_spec)
    store = TelemetryStore(settings.telemetry_db)
    state = ModelState()
    rng = np.random.default_rng()
    rollout: dict[str, Any] = {"mode": "champion_only", "status": "idle", "experiment_id": None}
    eval_lock = asyncio.Lock()

    # ------------------------------------------------------------ loading
    def _warmup(predictor: Predictor) -> None:
        """One dummy window end-to-end so numba/torch kernels compile before readiness."""
        import io

        import soundfile as sf

        buf = io.BytesIO()
        noise = np.random.default_rng(0).normal(0, 0.01, audio_spec.window_samples)
        sf.write(buf, noise.astype(np.float32), audio_spec.sample_rate, format="WAV")
        predictor.predict_proba(preprocess_call(buf.getvalue(), audio_spec, extractor))

    def _load_blocking(initial: bool) -> None:
        """Load @champion first: readiness (HTTP 200) depends on it alone. The optional
        @challenger is loaded afterwards so a slow or broken challenger can never delay
        or block the startup probe."""
        try:
            predictor, source, version = loader()
            _warmup(predictor)
            with state.lock:
                state.predictor, state.source, state.version = predictor, source, version
                state.status, state.error, state.ready_at = "ready", None, time.time()
            log.info("ready: champion=%s", version)
        except Exception as exc:
            log.exception("model load failed")
            with state.lock:
                if initial or state.predictor is None:
                    state.status, state.error = "failed", f"{type(exc).__name__}: {exc}"
                else:  # failed hot-reload: keep serving the previous model
                    state.error = f"reload failed: {exc}"
            state.reloading = False
            return
        try:
            chall = challenger_loader()
            if chall is not None:
                _warmup(chall[0])
        except Exception as exc:  # a broken challenger must never affect the champion
            log.warning("challenger not loaded: %s", exc)
            chall = None
        with state.lock:
            state.challenger = chall[0] if chall else None
            state.challenger_version = chall[2] if chall else None
        log.info("challenger: %s", state.challenger_version)
        state.reloading = False

    async def _reload_and_check(reason: str) -> None:
        before = (state.version, state.challenger_version)
        state.reloading = True
        await asyncio.to_thread(_load_blocking, False)
        after = (state.version, state.challenger_version)
        if before != after:
            await store.log_event(
                rollout.get("experiment_id"),
                "models_reloaded",
                {"reason": reason, "before": before, "after": after},
            )
            if rollout["mode"] != "champion_only":
                await _set_mode("champion_only", reason="models changed - experiment closed")

    async def _poll_aliases() -> None:
        """Course 7 slide 24: the serving process follows the registry aliases live."""
        while True:
            await asyncio.sleep(settings.poll_seconds)
            if state.source != "mlflow" or state.reloading:
                continue
            try:
                champ = await asyncio.to_thread(_registry_version, settings, "champion")
                chall = await asyncio.to_thread(_registry_version, settings, "challenger")
            except Exception as exc:
                log.warning("alias poll failed: %s", exc)
                continue
            if champ == chall:
                chall = None
            if champ != state.version or chall != state.challenger_version:
                log.info("alias change detected: champion=%s challenger=%s", champ, chall)
                await _reload_and_check("registry alias changed")

    # ------------------------------------------------------------ rollout
    async def _set_mode(
        mode: str,
        policy: str | None = None,
        baseline: float | None = None,
        mde: float | None = None,
        reason: str = "operator",
    ) -> dict[str, Any]:
        exp = None if mode == "champion_only" else f"{mode}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
        new: dict[str, Any] = {
            "experiment_id": exp,
            "mode": mode,
            "status": "running" if exp else "idle",
            "started_at": now_iso(),
            "champion_version": state.version,
            "challenger_version": state.challenger_version,
            "canary_step": 0,
            "challenger_percent": 0.0,
            "last_decision": None,
        }
        if mode == "canary":
            new["challenger_percent"] = float(rcfg["canary"]["steps"][0])
            new["steps"] = list(rcfg["canary"]["steps"])
        elif mode == "ab":
            ab = rcfg["ab"]
            base = baseline if baseline is not None else float(ab["baseline_rate"])
            eff = mde if mde is not None else float(ab["mde"])
            new.update(
                challenger_percent=50.0,
                baseline_rate=base,
                mde=eff,
                planned_n_per_arm=ro.ab_sample_size(base, eff, ab["alpha"], ab["power"]),
            )
        elif mode == "bandit":
            new["policy"] = policy or rcfg["bandit"]["policy"]
        rollout.clear()
        rollout.update(new)
        await store.save_rollout_state(rollout)
        await store.log_event(exp, f"mode_{mode}", {"reason": reason, **new})
        return rollout

    async def _route(call_id: str) -> tuple[str, float]:
        mode, status = rollout["mode"], rollout["status"]
        if state.challenger is None or mode in ("champion_only", "shadow"):
            return "champion", 1.0
        if mode in ("canary", "ab"):
            if status not in ("running", "complete"):
                return "champion", 1.0
            return ro.route_split(call_id, rollout["challenger_percent"])
        if mode == "bandit":
            counts = await store.bandit_counts(rollout["experiment_id"])
            succ = np.array([counts.get(a, (0, 0))[0] for a in ro.ARMS], float)
            pulls = np.array([counts.get(a, (0, 0))[1] for a in ro.ARMS], float)
            b = rcfg["bandit"]
            probs = bandits.policy_probs(
                rollout.get("policy", b["policy"]),
                succ,
                pulls,
                rng,
                float(b["epsilon"]),
                float(b["floor"]),
            )
            arm = bandits.choose(probs, rng)
            return ro.ARMS[arm], float(probs[arm])
        return "champion", 1.0

    async def _evaluate_canary() -> None:
        if rollout["mode"] != "canary" or rollout["status"] != "running":
            return
        async with eval_lock:
            exp, step = rollout["experiment_id"], rollout["canary_step"]
            calls = await store.experiment_calls(exp, step)
            scores = await store.window_scores_by_arm(exp, step)
            stats = ro.arm_stats(calls, scores)
            if stats["champion"]["calls"] < 10:  # 100% step: compare with earlier champion data
                all_calls = await store.experiment_calls(exp)
                all_scores = await store.window_scores_by_arm(exp)
                stats["champion"] = ro.arm_stats(all_calls, all_scores)["champion"]
                scores["champion"] = all_scores["champion"]
            result = ro.canary_decision(
                stats, scores["champion"], scores["challenger"], rcfg["canary"], step
            )
            rollout["last_decision"] = {**result, "step": step, "at": now_iso()}
            decision = result["decision"]
            if decision == "advance":
                rollout["canary_step"] = step + 1
                rollout["challenger_percent"] = float(rollout["steps"][step + 1])
                await store.log_event(
                    exp,
                    "canary_advanced",
                    {
                        "from_percent": rollout["steps"][step],
                        "to_percent": rollout["challenger_percent"],
                        "checks": result["checks"],
                    },
                )
            elif decision == "rollback":
                rollout["status"], rollout["challenger_percent"] = "rolled_back", 0.0
                await store.log_event(
                    exp,
                    "canary_rolled_back",
                    {"reason": result["reason"], "checks": result["checks"]},
                )
                log.warning("CANARY ROLLED BACK: %s", result["reason"])
            elif decision == "complete":
                rollout["status"] = "complete"
                await store.log_event(exp, "canary_complete", {"checks": result["checks"]})
            await store.save_rollout_state(rollout)

    # ------------------------------------------------------------ lifespan
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await store.init()
        saved = await store.load_rollout_state()
        if saved:
            rollout.update(saved)
        elif rcfg.get("mode", "champion_only") != "champion_only":
            rollout.update(mode=rcfg["mode"], status="pending")
        app.state.load_task = asyncio.create_task(asyncio.to_thread(_load_blocking, True))
        poll = asyncio.create_task(_poll_aliases()) if settings.poll_seconds > 0 else None
        yield
        if poll:
            poll.cancel()

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
    app.state.rollout = rollout

    def _health() -> HealthResponse:
        return HealthResponse(
            status=state.status,
            model_source=state.source,
            model_version=state.version,
            challenger_version=state.challenger_version,
            rollout_mode=rollout.get("mode"),
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
        app.state.reload_task = asyncio.create_task(_reload_and_check("manual reload"))
        return {"status": "reloading", "current_version": state.version}

    # ------------------------------------------------------------ predict
    def _score(predictor: Predictor, feats: dict) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        scores = np.clip(np.asarray(predictor.predict_proba(feats), dtype=float), 0.0, 1.0)
        if not np.all(np.isfinite(scores)):
            raise ValueError("model returned NaN/inf scores")
        return scores, (time.perf_counter() - t0) * 1000 / max(1, len(scores))

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: Request,
        background: BackgroundTasks,
        file: UploadFile = File(..., description="16 kHz mono WAV"),
        call_id: str | None = Form(None),
    ):
        champion, challenger = state.predictor, state.challenger
        if state.status != "ready" or champion is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "model_not_ready", "status": state.status},
                headers={"Retry-After": "5"},
            )
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="file too large (max 50 MB)")
        call_id = call_id or f"call-{uuid.uuid4().hex[:12]}"

        t0 = time.perf_counter()
        try:
            feats = await run_in_threadpool(preprocess_call, raw, audio_spec, extractor)
        except AudioContractError as exc:
            raise HTTPException(
                422, detail={"error": "audio_contract_violation", "message": str(exc)}
            ) from exc

        arm, propensity = await _route(call_id)
        error = 0
        served = champion if arm == "champion" or challenger is None else challenger
        try:
            scores, ms_per_window = await run_in_threadpool(_score, served, feats)
        except Exception:
            if served is champion:
                raise
            # Fallback pattern (Course 3): a failing challenger never fails the caller.
            log.exception("challenger failed on %s - falling back to champion", call_id)
            error = 1
            scores, ms_per_window = await run_in_threadpool(_score, champion, feats)
            served = champion
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
            call_id=call_id,
            received_at=now_iso(),
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
            model_version=str(served.version),
            model_source=str(state.source),
            latency_ms=round(latency_ms, 2),
            served_by=arm,
            rollout_mode=rollout["mode"],
        )
        meta = {
            "experiment_id": rollout.get("experiment_id"),
            "mode": rollout["mode"],
            "arm": arm,
            "propensity": round(propensity, 6),
            "canary_step": rollout.get("canary_step") if rollout["mode"] == "canary" else None,
            "model_ms_per_window": round(ms_per_window, 3),
            "error": error,
            "client": request.client.host if request.client else None,
        }
        shadow = challenger if rollout["mode"] == "shadow" and challenger is not None else None
        background.add_task(_after_call, resp, meta, shadow, feats)
        return resp

    async def _after_call(
        resp: PredictionResponse, meta: dict, shadow: Predictor | None, feats: dict
    ) -> None:
        """Background: audit row, then shadow scoring, then automatic canary analysis."""
        try:
            call: dict[str, Any] = resp.model_dump(exclude={"windows", "alert_segments"})
            call.update(n_alert_segments=len(resp.alert_segments), **meta)
            await store.log_call(call, [w.model_dump() for w in resp.windows])
        except Exception:
            log.exception("telemetry write failed for %s", resp.call_id)
            return
        if shadow is not None:
            try:
                s_scores, s_ms = await run_in_threadpool(_score, shadow, feats)
                s_alert = bool(
                    find_alert_segments(
                        s_scores,
                        feats["starts"],
                        feats["ends"],
                        settings.threshold,
                        settings.consecutive_windows,
                    )
                )
                await store.update_call(
                    resp.call_id,
                    shadow_version=shadow.version,
                    shadow_alert=int(s_alert),
                    shadow_max_score=float(s_scores.max()),
                    shadow_ms_per_window=round(s_ms, 3),
                    shadow_error=0,
                )
            except Exception:
                log.exception("shadow scoring failed for %s", resp.call_id)
                await store.update_call(
                    resp.call_id, shadow_error=1, shadow_version=getattr(shadow, "version", None)
                )
        if meta["mode"] == "canary":
            try:
                await _evaluate_canary()
            except Exception:
                log.exception("canary evaluation failed")

    # ------------------------------------------------------------ feedback
    @app.post("/feedback")
    async def feedback(body: FeedbackIn):
        """Dispatcher ground truth - the delayed reward of Course 4 (A/B, bandits)."""
        call = None
        for _ in range(30):  # the audit row is written just after the response
            call = await store.get_call(body.call_id)
            if call is not None:
                break
            await asyncio.sleep(0.1)
        if call is None:
            raise HTTPException(404, detail="unknown call_id")
        truth = body.true_label == "stressed"
        fields: dict[str, Any] = {
            "true_label": body.true_label,
            "correct": int(bool(call["alert"]) == truth),
            "feedback_at": now_iso(),
        }
        if call.get("shadow_alert") is not None:
            fields["shadow_correct"] = int(bool(call["shadow_alert"]) == truth)
        await store.update_call(body.call_id, **fields)
        return {"call_id": body.call_id, **fields}

    # ------------------------------------------------------------ rollout API
    @app.get("/rollout")
    async def get_rollout():
        return {
            **rollout,
            "champion_version": state.version,
            "challenger_version": state.challenger_version,
        }

    @app.post("/rollout")
    async def set_rollout(body: RolloutIn):
        if body.mode != "champion_only" and state.challenger is None:
            raise HTTPException(
                409,
                detail="no @challenger model loaded (it must differ from "
                "@champion); train one and POST /admin/reload",
            )
        return await _set_mode(body.mode, body.policy, body.baseline_rate, body.mde)

    @app.get("/rollout/experiments")
    async def rollout_experiments():
        return await store.experiments()

    @app.get("/rollout/events")
    async def rollout_events(limit: int = 100):
        return await store.events(min(max(limit, 1), 1000))

    @app.get("/rollout/report")
    async def rollout_report(experiment_id: str | None = None):
        exp = experiment_id or rollout.get("experiment_id")
        if not exp:
            return {"rollout": rollout, "report": None}
        calls = await store.experiment_calls(exp)
        scores = await store.window_scores_by_arm(exp)
        stats = ro.arm_stats(calls, scores)
        mode = calls[0]["mode"] if calls else rollout["mode"]
        out: dict[str, Any] = {
            "is_current": exp == rollout.get("experiment_id"),
            "rollout": rollout,
            "experiment_id": exp,
            "mode": mode,
            "arms": stats,
            "n_calls": len(calls),
        }
        out["timeline"] = [
            {
                "i": i,
                "arm": c["arm"],
                "propensity": c["propensity"],
                "step": c["canary_step"],
                "correct": c["correct"],
                "error": c["error"],
            }
            for i, c in enumerate(calls)
        ]
        if mode == "shadow":
            out["shadow"] = ro.shadow_report(calls)
        if mode == "canary":
            per_step = []
            for step in sorted({c["canary_step"] for c in calls if c["canary_step"] is not None}):
                rows = [c for c in calls if c["canary_step"] == step]
                per_step.append(
                    {
                        "step": step,
                        "percent": rcfg["canary"]["steps"][step],
                        "calls": len(rows),
                        "challenger_calls": sum(c["arm"] == "challenger" for c in rows),
                    }
                )
            last_decision = rollout.get("last_decision")
            if exp != rollout.get("experiment_id"):  # a past experiment: read the audit trail
                ev = await store.last_event(exp, "canary_")
                last_decision = None
                if ev and ev["details"]:
                    kind = {"canary_rolled_back": "rollback", "canary_complete": "complete"}.get(
                        ev["event"], "advance"
                    )
                    default_reason = {
                        "complete": "100% reached with healthy guardrails",
                        "advance": "healthy - advanced",
                    }.get(kind, ev["event"])
                    last_decision = {
                        "decision": kind,
                        "at": ev["at"],
                        "reason": ev["details"].get("reason", default_reason),
                        "checks": ev["details"].get("checks", {}),
                        "step": max(c["canary_step"] or 0 for c in calls),
                    }
            if exp == rollout.get("experiment_id"):
                status, cur_step = rollout.get("status"), rollout.get("canary_step", 0)
            else:
                kind = (last_decision or {}).get("decision")
                status = {"rollback": "rolled_back", "complete": "complete"}.get(kind, "stopped")
                cur_step = (last_decision or {}).get("step", 0)
            out["canary"] = {
                "status": status,
                "current_step": cur_step,
                "steps": list(rcfg["canary"]["steps"]),
                "per_step": per_step,
                "last_decision": last_decision,
                "psi_all": ro.psi(scores["champion"], scores["challenger"]),
            }
        if mode == "ab":
            planned = rollout.get("planned_n_per_arm")
            if exp != rollout.get("experiment_id"):
                ev = await store.last_event(exp, "mode_ab")
                planned = (ev or {}).get("details", {}).get("planned_n_per_arm")
            out["ab"] = ro.ab_report(stats, int(planned or 0), float(rcfg["ab"]["alpha"]))
        if mode == "bandit":
            acc = {a: stats[a]["accuracy"] for a in ro.ARMS}
            if all(v is not None for v in acc.values()):
                best = max(acc.values())
                labelled = [c for c in calls if c["correct"] is not None]
                regret = sum(best - acc[c["arm"]] for c in labelled)
                gap = abs(acc["champion"] - acc["challenger"])
                out["bandit"] = {
                    "estimated_regret": regret,
                    "ab_test_equivalent_regret": gap * len(labelled) / 2,
                    "policy": rollout.get("policy"),
                    "accuracy": acc,
                }
        return out

    # ------------------------------------------------------------ telemetry
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


app = create_app()
