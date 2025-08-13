from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import generate_latest

from app.api.routers import router as api_router
from app.core.config import settings
from app.middleware.request_id import RequestIDMiddleware
from app.core.logging import setup_logging, get_logger
from app.core.metrics import MODEL_LOADED_INFO, MODEL_LOAD_FAILED_TOTAL
from app.models.loader import ModelLoader
from app.models.store import model_store

logger = get_logger(__name__)


MODELS_TO_LOAD = {
    "lightgbm": "FraudDetector-lightgbm",
    "xgboost": "FraudDetector-xgboost",
    "logreg": "FraudDetector-logistic_regression",
}

# --- App State ---
# A simple dict to hold application state.
# 'models_ready' will be updated by the model loading service.
app_state = {"models_ready": False}


def _load_models() -> dict:
    """Load all registered models and return them as a dictionary of bundles."""
    loader = ModelLoader()
    loaded_bundles = {}
    for key, name in MODELS_TO_LOAD.items():
        try:
            bundle = loader.load_model_bundle(key, name)
            loaded_bundles[key] = bundle
        except Exception as e:
            logger.error("Failed to load model", model_name=name, exc_info=e)
            MODEL_LOAD_FAILED_TOTAL.labels(model_name=name).inc()
    return loaded_bundles


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context.
    - Sets up logging on startup.
    - Loads ML models on startup.
    """
    setup_logging()
    logger.info(
        "Starting application",
        git_sha=settings.GIT_SHA,
        build_time=settings.BUILD_TIME,
    )

    loaded_bundles = _load_models()
    if len(loaded_bundles) == len(MODELS_TO_LOAD):
        model_store.replace_all(loaded_bundles)
        for bundle in loaded_bundles.values():
            MODEL_LOADED_INFO.labels(
                model_name=bundle["name"],
                model_version=bundle["version"],
                model_stage=bundle["stage"],
                run_id=bundle["run_id"],
            ).set(1)
        app_state["models_ready"] = True
        logger.info("All models loaded successfully and store is ready.")
    else:
        logger.warning("Service is starting with some models not loaded.")

    yield
    logger.info("Shutting down application")


app = FastAPI(
    title="Fraud Inference API",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Middleware ---
app.add_middleware(RequestIDMiddleware)

# --- Health Check Endpoints ---
@app.get("/livez", tags=["Health"])
async def livez():
    """
    Liveness probe. Always returns 200 OK to indicate the process is running.
    """
    return {"status": "ok"}


@app.get("/readyz", tags=["Health"])
async def readyz():
    """
    Readiness probe. Returns 200 OK if the models are loaded and ready.
    """
    if app_state["models_ready"]:
        return {"status": "ok"}
    else:
        return Response(
            content='{"status": "not_ready"}',
            status_code=503,
            media_type="application/json",
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(), media_type="text/plain")


# --- API Routers ---
app.include_router(api_router, prefix="/v1")
