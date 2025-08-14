import time
import json
from typing import Callable, TypedDict

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel
from mlflow.exceptions import MlflowException

from app.core.logging import get_logger
from app.core.metrics import MODEL_WARMUP_LATENCY_SECONDS, MODEL_LOAD_SUCCESS_TOTAL
from app.models.registry import MlflowModelSelector
from app.services.validator import DependencyValidator

logger = get_logger(__name__)

# --- Data Structures ---


class ModelBundle(TypedDict):
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
    unwrapped_model = model
    if hasattr(model, "unwrap_python_model"):
        try:
            unwrapped_model = model.unwrap_python_model()
        except MlflowException:
            logger.warning(
                "Failed to unwrap underlying model; using original pyfunc model",
                model_class=type(model).__name__,
                exc_info=True,
            )

    if hasattr(unwrapped_model, "predict_proba"):
        logger.info("Using 'predict_proba' for model prediction.")
        return lambda df: unwrapped_model.predict_proba(df)[:, 1]
    if hasattr(unwrapped_model, "decision_function"):
        logger.info("Using 'decision_function' for model prediction.")
        return lambda df: unwrapped_model.decision_function(df)
    logger.info("Using 'predict' for model prediction.")
    return lambda df: unwrapped_model.predict(df)


def _ensure_parsed_signature(sig: dict) -> dict:
    if not isinstance(sig, dict):
        raise ValueError("Signature must be a dict after to_dict().")

    def maybe_parse(x):
        if isinstance(x, str):
            try:
                return json.loads(x)  # will turn your JSON string into list[dict]
            except Exception:
                return x
        return x

    sig["inputs"] = maybe_parse(sig.get("inputs"))
    sig["outputs"] = maybe_parse(sig.get("outputs"))
    sig["params"] = maybe_parse(sig.get("params"))
    return sig


def _create_warmup_payload(signature: dict, n_rows: int = 1) -> pd.DataFrame:
    signature = _ensure_parsed_signature(signature)

    inputs = signature.get("inputs")
    if inputs is None:
        raise ValueError(
            "Model signature has no 'inputs' key, cannot create warmup payload."
        )

    if isinstance(inputs, dict):
        inputs = [inputs]
    if not isinstance(inputs, list):
        raise ValueError(f"Unrecognized signature['inputs'] type: {type(inputs)}")

    data: dict[str, list[int | float | str | bool]] = {}
    for i, spec in enumerate(inputs):
        if not isinstance(spec, dict):
            name, mlt = f"f{i}", "double"
        else:
            name = spec.get("name") or f"f{i}"
            mlt = (spec.get("type") or "string").lower()

        if mlt in ("long", "integer", "int"):
            val = 0
        elif mlt in ("float", "double"):
            val = 0.0
        elif mlt in ("boolean", "bool"):
            val = False
        else:
            val = ""
        data[name] = [val] * n_rows

    return pd.DataFrame(data)


# --- Model Loader ---


class ModelLoader:
    def __init__(self):
        self.selector = MlflowModelSelector()
        self.validator = DependencyValidator()

    def load_model_bundle(self, key: str, model_name: str) -> ModelBundle:
        logger.info(
            "Starting model loading process", model_key=key, model_name=model_name
        )

        # 1) Select model
        model_info = self.selector.select_model_version(model_name)
        if not model_info:
            raise RuntimeError(f"Could not select a version for model '{model_name}'")
        logger.info(
            "Selected model version", version=model_info.version, stage=model_info.stage
        )

        # 2) Validate env
        self.validator.validate(model_info.source_uri)
        logger.info("Dependency validation passed")

        # 3) Load
        model = mlflow.pyfunc.load_model(model_info.source_uri)
        logger.info("Model loaded from MLflow", uri=model_info.source_uri)

        # 4) Predict fn
        predict_fn = _pick_predict_fn(model)

        # 5) Warmup
        sig_obj = getattr(
            model.metadata, "get_signature", lambda: model.metadata.signature
        )()
        signature = sig_obj.to_dict() if hasattr(sig_obj, "to_dict") else sig_obj
        logger.info("Signature (raw)", signature=signature)

        warmup_payload = _create_warmup_payload(signature, n_rows=1)
        logger.info(
            "Warming up model with sample payload...",
            payload_cols=list(warmup_payload.columns),
        )

        start_time = time.monotonic()
        try:
            predict_fn(warmup_payload)
        except TypeError:
            predict_fn(warmup_payload.to_dict(orient="list"))
        warmup_latency = time.monotonic() - start_time

        MODEL_WARMUP_LATENCY_SECONDS.labels(
            model_name=model_name, model_version=model_info.version
        ).observe(warmup_latency)
        logger.info("Model warmup complete", latency_sec=round(warmup_latency, 4))

        # 6) Metric
        stage_value = getattr(
            model_info.stage, "value", str(model_info.stage)
        )  # stage may be None
        MODEL_LOAD_SUCCESS_TOTAL.labels(
            model_name=model_name,
            model_version=model_info.version,
            model_stage=stage_value,
        ).inc()

        # 7) Bundle
        bundle = ModelBundle(
            key=key,
            name=model_name,
            version=model_info.version,
            stage=stage_value,
            run_id=model_info.run_id,
            uri=model_info.source_uri,
            signature=_ensure_parsed_signature(signature),
            model=model,
            predict_fn=predict_fn,
        )
        logger.info("Successfully created model bundle", model_key=key)
        return bundle
