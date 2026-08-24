"""FastAPI app entrypoint. Verifies the DB is reachable and creates tables
on startup (failing loudly, not hanging, if it can't), starts the daily
bandit-update scheduler, and normalizes every error response — including
FastAPI's own validation errors and any unhandled exception — to
`{"error": ..., "code": ...}` with no stack trace ever exposed.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text

from api.errors import APIError
from api.routes import accounts, attribution_baselines, bandit_state, outcomes, recovery
from bandit.updater import start_scheduler
from db.session import engine, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.main")

_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("startup_db_connection_failed error=%r", exc)
        raise RuntimeError(f"cannot connect to the database: {exc}") from exc

    init_db()
    _scheduler = start_scheduler()
    logger.info("startup_complete")

    yield

    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
    logger.info("shutdown_complete")


app = FastAPI(title="AI Revenue Recovery System", lifespan=lifespan)

app.include_router(accounts.router)
app.include_router(recovery.router)
app.include_router(outcomes.router)
app.include_router(bandit_state.router)
app.include_router(attribution_baselines.router)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message, "code": exc.code})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": str(exc.errors()), "code": "validation_error"})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception path=%s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal server error", "code": "internal_error"})
