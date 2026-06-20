"""FLOWTOX_REGIME_01 backend API.

Wraps the quantitative strategy engine with endpoints for instruments, single
backtests, walk-forward optimization (background jobs), run retrieval, and
downloadable artifacts.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.cors import CORSMiddleware

import strategy_service as svc

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

app = FastAPI(title="FLOWTOX_REGIME_01 API")
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("flowtox")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str = "ES"
    toxic_continuation_threshold: float = Field(0.55, ge=0.0, le=1.0)
    toxic_reversal_threshold: float = Field(0.55, ge=0.0, le=1.0)
    max_hold_bars: int = Field(15, ge=1, le=200)
    regime_exit_enabled: bool = True
    drawdown_method: str = Field("anchored", pattern="^(anchored|spec)$")
    regime_model: str = Field("gmm", pattern="^(gmm|hmm)$")


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str = "ES"
    drawdown_method: str = Field("anchored", pattern="^(anchored|spec)$")
    regime_model: str = Field("gmm", pattern="^(gmm|hmm)$")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@api_router.get("/")
async def root():
    return {"service": "FLOWTOX_REGIME_01", "status": "ok"}


@api_router.get("/instruments")
async def instruments():
    return {"instruments": svc.list_instruments()}


@api_router.get("/strategy/info")
async def strategy_info():
    return svc.strategy_info()


@api_router.get("/model/status")
async def model_status(symbol: str = "ES"):
    return {"symbol": symbol.upper(), "ready": svc.is_model_ready(symbol)}


@api_router.post("/model/warm")
async def model_warm(req: OptimizeRequest):
    """Fit + cache models for a symbol (first call is slow ~60s)."""
    try:
        cache = await run_in_threadpool(svc.ensure_models, req.symbol, req.regime_model)
        return {"symbol": req.symbol.upper(), "ready": True, "model_info": cache["model_info"]}
    except Exception as exc:  # noqa: BLE001
        logger.exception("model_warm failed")
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.post("/backtest/single")
async def backtest_single(req: BacktestRequest):
    params = {
        "toxic_continuation_threshold": float(req.toxic_continuation_threshold),
        "toxic_reversal_threshold": float(req.toxic_reversal_threshold),
        "max_hold_bars": int(req.max_hold_bars),
        "regime_exit_enabled": bool(req.regime_exit_enabled),
    }
    try:
        payload = await run_in_threadpool(svc.run_single, req.symbol, params, req.drawdown_method, req.regime_model)
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.exception("backtest_single failed")
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.post("/optimize/start")
async def optimize_start(req: OptimizeRequest):
    try:
        job_id = svc.start_optimization(req.symbol, req.drawdown_method, req.regime_model)
        return {"job_id": job_id, "status": "queued"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("optimize_start failed")
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.get("/pine-script")
async def pine_script():
    """Download the TradingView Pine Script port of the strategy."""
    path = "/app/strategy_01_flowtox_regime/FLOWTOX_REGIME_01.pine"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Pine script not found")
    return FileResponse(path, media_type="text/plain", filename="FLOWTOX_REGIME_01.pine")


@api_router.get("/optimize/status/{job_id}")
async def optimize_status(job_id: str):
    job = svc.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@api_router.get("/runs/{run_id}")
async def get_run(run_id: str):
    payload = svc.get_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return payload


@api_router.get("/runs/{run_id}/download/{kind}")
async def download_run(run_id: str, kind: str):
    path = svc.run_file_path(run_id, kind)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    filename = os.path.basename(path)
    media = "application/json" if kind == "metrics" else "text/csv"
    return FileResponse(path, media_type=media, filename=filename)


app.include_router(api_router)


@app.on_event("startup")
async def _warm_default_models():
    """Pre-warm the default instrument (ES) for both regime models in the
    background so the first user backtest is instant (avoids cold-start gateway
    timeouts). Non-blocking; failures are logged and ignored."""
    def _warm():
        for rm in ("gmm", "hmm"):
            try:
                svc.ensure_models("ES", rm)
                logger.info("Pre-warmed ES:%s", rm)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pre-warm ES:%s failed: %s", rm, exc)
    try:
        svc._EXECUTOR.submit(_warm)
    except Exception:  # noqa: BLE001
        pass

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
