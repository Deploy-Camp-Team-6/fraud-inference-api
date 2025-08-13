from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, AliasChoices, model_validator
from typing_extensions import Annotated


# ---- Enums / simple types ----
ModelKey = Literal["lightgbm", "xgboost", "logreg"]

# guardrails for batch sizes & payload size (adjust via config if needed)
MaxBatch = Annotated[int, Field(ge=1, le=10_000)]


# ---- Request models ----
class PredictRow(BaseModel):
    """
    A single feature row. Fields are populated dynamically at runtime from
    the MLflow signature (validator layer), so this serves as a permissive
    container here. We still forbid unknown keys at the *validated* layer.
    """
    model_config = ConfigDict(extra="allow")  # schema is enforced later by the signature validator


class PredictRequest(BaseModel):
    """
    Accept either:
      - a single object: { "model": "...", "features": {...} }
      - a batch:          { "model": "...", "instances": [ {...}, {...} ] }
    Optionally accept a threshold and a client-supplied request_id.
    """
    model: ModelKey
    # allow either "features" or legacy "instance"
    features: Optional[PredictRow] = Field(
        default=None,
        validation_alias=AliasChoices("features", "instance"),
        description="Single-row feature payload."
    )
    instances: Optional[List[PredictRow]] = Field(
        default=None,
        description="Batch of feature rows; order is preserved."
    )
    threshold: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default=None, description="Optional decision threshold; defaults to service config."
    )
    request_id: Optional[UUID] = Field(
        default=None, description="Client-supplied correlation id; server will generate if absent."
    )

    @model_validator(mode="after")
    def _exactly_one_payload(cls, values: "PredictRequest") -> "PredictRequest":
        has_single = values.features is not None
        has_batch = values.instances is not None and len(values.instances) > 0
        if has_single == has_batch:
            # both provided or both missing
            raise ValueError('Provide exactly one of "features" or "instances (non-empty)".')
        if has_batch:
            if len(values.instances) > 10_000:
                raise ValueError("Batch size exceeds the maximum of 10_000 rows.")
        return values

    def is_batch(self) -> bool:
        return self.instances is not None


# ---- Response models ----
class PredictionMeta(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str
    model_version: str
    model_stage: str
    run_id: str
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Server-side UTC timestamp."
    )
    latency_ms: float


class SinglePrediction(BaseModel):
    # Returned for single-row inference
    prediction: Union[int, float, bool]
    score: Optional[float] = Field(default=None, description="Probability/score if available.")
    threshold: Optional[float] = None


class BatchPrediction(BaseModel):
    # Returned for batch inference
    predictions: List[Union[int, float, bool]]
    scores: Optional[List[float]] = Field(default=None, description="Per-row scores if available.")
    threshold: Optional[float] = None
    errors: Optional[List[Optional[str]]] = Field(
        default=None, description="Per-row error strings (None if ok), order-preserving."
    )


class PredictResponse(BaseModel):
    meta: PredictionMeta
    result: Union[SinglePrediction, BatchPrediction]


# ---- Error envelope (FastAPI error handler can use this) ----
class ErrorDetail(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str


class ErrorResponse(BaseModel):
    error: str
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[UUID] = None
