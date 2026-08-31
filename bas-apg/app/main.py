"""
BAS-APG — FastAPI Application Entry Point

Initializes backend + ML subsystems only:
  Camera, YOLO, MediaPipe, HOI Tracker, FSM, TTS Voice

NO frontend, NO database. Pure backend + ML pipeline.

Usage:
    conda activate bas_apg_env
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.core.camera import CameraBuffer
from app.core.config import get_settings
from app.core.ws_manager import WSManager
from app.engines.hoi_tracker import HOITracker
from app.engines.procedure_fsm import ProcedureFSM
from app.engines.recovery_engine import RecoveryEngine
from app.engines.voice_alert import BackgroundTTSWorker
from app.routers import api as api_module
from app.routers.api import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    print("=" * 60)
    print("  BAS AI Procedure Guardian — FastAPI Started")
    print("  (Running in Decoupled Worker Mode via Watchdog)")
    print("=" * 60)

    # --- 1. WebSocket Manager ---
    print("[INIT] Creating WebSocket manager...")
    ws_mgr = WSManager()
    api_module.ws_manager = ws_mgr

    yield  # ========== APP IS RUNNING ==========

    # --- SHUTDOWN ---
    print("\n[SHUTDOWN] API Server stopping...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title="BAS AI Procedure Guardian",
        description=(
            "A.T.L.A.S. — Backend + ML pipeline for offline AI procedure monitoring. "
            "Camera → YOLO → MediaPipe → HOI → FSM → Voice alerts."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    application.include_router(router)
    return application


app = create_app()
