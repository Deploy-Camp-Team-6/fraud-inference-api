from fastapi import APIRouter, Depends, HTTPException
from typing import Dict

from app.core.config import settings
from app.models.store import model_store
from app.schemas.predict import (
    PredictRequest,
    PredictResponse,
)
from app.services.infer import InferenceService


router = APIRouter()


def get_inference_service():
    """Dependency injector for the inference service."""
    return InferenceService()


@router.post(
    "/admin/refresh-models", tags=["Admin"]
)
async def refresh_models() -> Dict[str, Dict[str, str]]:
    """
    Reloads all models from the registry, providing a hot-swap.
    Returns a diff of the model versions that were changed.
    """
    old_bundles = model_store.snapshot()

    # This should be run in a background thread in a real-world scenario
    # to avoid blocking the server.
    from app.main import _load_models

    new_bundles = _load_models()

    if not new_bundles:
        raise HTTPException(
            status_code=500, detail="Failed to load any models during refresh."
        )

    model_store.replace_all(new_bundles)

    # Compute diff
    diff = {}
    for key, new_bundle in new_bundles.items():
        old_bundle = old_bundles.get(key)
        if not old_bundle or old_bundle["version"] != new_bundle["version"]:
            diff[new_bundle["name"]] = {
                "from": old_bundle["version"] if old_bundle else "None",
                "to": new_bundle["version"],
            }
    return diff


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    request: PredictRequest,
    inference_service: InferenceService = Depends(get_inference_service),
):
    """
    Performs inference using one of the loaded fraud detection models.
    """
    model_key = request.model
    bundle = model_store.get(model_key)
    if not bundle:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_key}' not found or not ready.",
        )

    try:
        response = inference_service.predict(bundle, request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@router.get("/version", tags=["Health"])
async def version():
    """
    Returns service version and loaded model information.
    """
    loaded_models = {}
    for key, bundle in model_store.snapshot().items():
        loaded_models[key] = {
            "name": bundle["name"],
            "version": bundle["version"],
            "stage": bundle["stage"],
            "run_id": bundle["run_id"],
            "signature_inputs": [col["name"] for col in bundle["signature"]["inputs"]],
        }

    return {
        "build_time": settings.BUILD_TIME,
        "git_sha": settings.GIT_SHA,
        "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
        "loaded_models": loaded_models,
    }
