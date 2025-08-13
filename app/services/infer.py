import time
import pandas as pd
from uuid import uuid4
from typing import List, Dict, Any, Union

from app.core.config import settings
from app.models.loader import ModelBundle
from app.schemas.predict import (
    PredictRequest,
    PredictResponse,
    PredictionMeta,
    SinglePrediction,
    BatchPrediction,
)

class InferenceService:
    """
    Encapsulates the core logic for model inference.
    """

    def predict(self, bundle: ModelBundle, request: PredictRequest) -> PredictResponse:
        """
        Performs inference on a single instance or a batch of instances.
        """
        start_time = time.monotonic()

        # 1. Prepare DataFrame from request
        df = self._create_dataframe(request)

        # 2. Validate DataFrame against model signature
        validated_df, errors = self._validate_dataframe(df, bundle["signature"])

        # 3. Perform inference
        scores = bundle["predict_fn"](validated_df) if not validated_df.empty else []

        # 4. Apply threshold and assemble results
        threshold = request.threshold or settings.SERVICE_THRESHOLD

        if request.is_batch():
            result = self._create_batch_result(scores, errors, threshold)
        else:
            # For single prediction, if validation failed, the result is an error.
            # This logic can be refined, but for now we assume it passes or fails as one.
            score = scores[0] if len(scores) > 0 else None
            result = self._create_single_result(score, threshold)

        # 5. Assemble final response
        latency_ms = (time.monotonic() - start_time) * 1000
        meta = PredictionMeta(
            model_name=bundle["name"],
            model_version=bundle["version"],
            model_stage=bundle["stage"],
            run_id=bundle["run_id"],
            request_id=request.request_id or uuid4(),
            latency_ms=latency_ms,
        )
        return PredictResponse(meta=meta, result=result)

    def _create_dataframe(self, request: PredictRequest) -> pd.DataFrame:
        """Creates a pandas DataFrame from the prediction request."""
        if request.is_batch():
            return pd.DataFrame([row.model_dump() for row in request.instances])
        else:
            return pd.DataFrame([request.features.model_dump()])

    def _validate_dataframe(self, df: pd.DataFrame, signature: dict) -> tuple[pd.DataFrame, list | None]:
        """
        Validates the DataFrame against the model's signature.
        - Enforces column presence and order.
        - Drops unknown columns if not allowed.
        - TODO: Add type coercion and more robust per-row error handling.
        """
        expected_cols = [col["name"] for col in signature.get("inputs", [])]

        # For now, a simple validation. This can be expanded.
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {list(missing_cols)}")

        if not settings.ALLOW_EXTRA:
            unknown_cols = set(df.columns) - set(expected_cols)
            if unknown_cols:
                df = df.drop(columns=list(unknown_cols))

        # Ensure correct column order
        return df[expected_cols], None

    def _create_single_result(self, score: float | None, threshold: float) -> SinglePrediction:
        """Creates a SinglePrediction object."""
        if score is None:
             # This case happens if validation fails for the single row.
             # A more robust implementation would have specific error handling here.
            return SinglePrediction(prediction=False, score=None, threshold=threshold)

        prediction = 1 if score >= threshold else 0
        return SinglePrediction(prediction=prediction, score=score, threshold=threshold)

    def _create_batch_result(self, scores: list, errors: list | None, threshold: float) -> BatchPrediction:
        """Creates a BatchPrediction object."""
        # This is a simplified version. A real implementation would map scores to non-error rows.
        predictions = [(1 if s >= threshold else 0) for s in scores]
        return BatchPrediction(predictions=predictions, scores=scores, threshold=threshold, errors=errors)
