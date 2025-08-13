import time
from typing import Callable, TypedDict

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel

from app.core.logging import get_logger
from app.core.metrics import MODEL_WARMUP_LATENCY_SECONDS, MODEL_LOAD_SUCCESS_TOTAL
from app.models.registry import MlflowModelSelector
from app.services.validator import DependencyValidator

logger = get_logger(__name__)

# --- Data Structures ---


class ModelBundle(TypedDict):
    """A dictionary containing all artifacts for a loaded model."""

    key: str
    name: str
    version: str
    stage: str
    run_id: str
    uri: str
    signature: dict
    model: PyFuncModel
    predict_fn: Callable[[pd.DataFrame], np.ndarray | pd.Series]


# --- Helper Functions ---


def _pick_predict_fn(
    model: PyFuncModel,
) -> Callable[[pd.DataFrame], np.ndarray | pd.Series]:
    """
    Selects the appropriate prediction function from a loaded model.
    Prefers probability scores for classifiers, otherwise falls back to predict.
    """
    # The actual model is often wrapped, e.g., in `model.models` for pipelines
    unwrapped_model = (
        model.unwrap_python_model() if hasattr(model, "unwrap_python_model") else model
    )

    if hasattr(unwrapped_model, "predict_proba"):
        logger.info("Using 'predict_proba' for model prediction.")
        # Return the probability of the positive class (class 1)
        return lambda df: unwrapped_model.predict_proba(df)[:, 1]

    if hasattr(unwrapped_model, "decision_function"):
        logger.info("Using 'decision_function' for model prediction.")
        return lambda df: unwrapped_model.decision_function(df)

    logger.info("Using 'predict' for model prediction.")
    return lambda df: unwrapped_model.predict(df)


def _create_warmup_payload(signature: dict) -> pd.DataFrame:
    """Creates a minimal valid payload from the model signature to warm up the model."""
    inputs = signature.get("inputs", [])
    if not inputs:
        raise ValueError("Model signature has no inputs, cannot create warmup payload.")

    data = {}
    for inp in inputs:
        name = inp["name"]
        dtype = inp["type"]
        # Map MLflow types to numpy dtypes and create dummy data
        if dtype in ["long", "integer"]:
            data[name] = [0]
        elif dtype in ["float", "double"]:
            data[name] = [0.0]
        else:  # string, binary, etc.
            data[name] = [""]

    return pd.DataFrame(data)


# --- Model Loader ---


class ModelLoader:
    """
    Handles the full lifecycle of selecting, validating, loading, and warming up a model.
    """

    def __init__(self):
        self.selector = MlflowModelSelector()
        self.validator = DependencyValidator()

    def load_model_bundle(self, key: str, model_name: str) -> ModelBundle:
        """
        Orchestrates the model loading process.
        """
        logger.info(
            "Starting model loading process", model_key=key, model_name=model_name
        )

        # 1. Select model version
        model_info = self.selector.select_model_version(model_name)
        if not model_info:
            raise RuntimeError(f"Could not select a version for model '{model_name}'")
        logger.info(
            "Selected model version", version=model_info.version, stage=model_info.stage
        )

        # 2. Validate dependencies
        self.validator.validate(model_info.source_uri)
        logger.info("Dependency validation passed")

        # 3. Load model from MLflow
        model = mlflow.pyfunc.load_model(model_info.source_uri)
        logger.info("Model loaded from MLflow", uri=model_info.source_uri)

        # 4. Normalize prediction function
        predict_fn = _pick_predict_fn(model)

        # 5. Warmup
        signature = model.metadata.get_signature()
        if not signature:
            raise ValueError("Model is missing a signature.")

        warmup_payload = _create_warmup_payload(signature.to_dict())

        logger.info(
            "Warming up model with sample payload...",
            payload_cols=list(warmup_payload.columns),
        )
        start_time = time.monotonic()
        predict_fn(warmup_payload)
        end_time = time.monotonic()
        warmup_latency = end_time - start_time

        MODEL_WARMUP_LATENCY_SECONDS.labels(
            model_name=model_name, model_version=model_info.version
        ).observe(warmup_latency)
        logger.info("Model warmup complete", latency_sec=round(warmup_latency, 4))

        # 6. Record success metric
        MODEL_LOAD_SUCCESS_TOTAL.labels(
            model_name=model_name,
            model_version=model_info.version,
            model_stage=model_info.stage.value,
        ).inc()

        # 7. Assemble and return bundle
        bundle = ModelBundle(
            key=key,
            name=model_name,
            version=model_info.version,
            stage=model_info.stage.value,
            run_id=model_info.run_id,
            uri=model_info.source_uri,
            signature=signature.to_dict(),
            model=model,
            predict_fn=predict_fn,
        )
        logger.info("Successfully created model bundle", model_key=key)
        return bundle
