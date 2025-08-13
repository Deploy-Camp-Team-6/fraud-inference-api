from enum import Enum
from operator import attrgetter
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
import os

logger = get_logger(__name__)

# --- Enums and Schemas ---


class ModelStage(str, Enum):
    """
    Enum for MLflow model stages. Order defines preference.
    """

    PRODUCTION = "Production"
    STAGING = "Staging"
    NONE = "None"

    @classmethod
    def from_str(cls, stage: str | None) -> "ModelStage":
        if stage is None:
            return cls.NONE
        try:
            return cls(stage.title())
        except ValueError:
            return cls.NONE


class ModelInfo(BaseModel):
    """
    Pydantic model to store metadata of a selected model version.
    """

    name: str
    version: str
    stage: ModelStage = Field(alias="current_stage")
    run_id: str
    source_uri: str = Field(..., alias="source")
    last_updated_timestamp: int


# --- Model Selector ---


class MlflowModelSelector:
    """
    Selects the best version of a registered model from MLflow based on a set of rules.
    """

    def __init__(self):
        self._set_mlflow_env()
        self.client = mlflow.tracking.MlflowClient(
            tracking_uri=settings.MLFLOW_TRACKING_URI
        )
        self.stage_preference = [
            ModelStage.PRODUCTION,
            ModelStage.STAGING,
            ModelStage.NONE,
        ]

    def _set_mlflow_env(self) -> None:
        """Set MLflow authentication and storage environment variables."""
        if settings.MLFLOW_TRACKING_USERNAME and settings.MLFLOW_TRACKING_PASSWORD:
            os.environ["MLFLOW_TRACKING_USERNAME"] = settings.MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = (
                settings.MLFLOW_TRACKING_PASSWORD.get_secret_value()
            )
        if settings.MLFLOW_S3_ENDPOINT_URL:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
            os.environ["AWS_SECRET_ACCESS_KEY"] = (
                settings.AWS_SECRET_ACCESS_KEY.get_secret_value()
            )
        if settings.AWS_DEFAULT_REGION:
            os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION

    def _model_version_to_dict(self, version: ModelVersion) -> dict[str, Any]:
        """Safely convert a MLflow ModelVersion object to a dictionary."""
        try:
            return version.to_dict()  # type: ignore[no-any-return]
        except AttributeError:
            attrs = {k.lstrip("_"): v for k, v in vars(version).items()}
            keys = [
                "name",
                "version",
                "current_stage",
                "run_id",
                "source",
                "last_updated_timestamp",
            ]
            return {k: attrs.get(k) for k in keys}

    def select_model_version(
        self, model_name: str, champion_alias: str = "champion"
    ) -> ModelInfo | None:
        """
        Selects the best model version according to the rules:
        1. Alias 'champion' if it exists.
        2. Stage preference: Production > Staging > None.
        3. Most recent last_updated_timestamp as a tie-breaker.
        """
        logger.info("Starting model selection", model_name=model_name)

        # 1. Check for 'champion' alias
        try:
            champion_version = self.client.get_model_version_by_alias(
                model_name, champion_alias
            )
            if champion_version:
                logger.info(
                    "Selected model version by alias",
                    model_name=model_name,
                    alias=champion_alias,
                    version=champion_version.version,
                )
                return ModelInfo.model_validate(
                    self._model_version_to_dict(champion_version)
                )
        except mlflow.exceptions.RestException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                logger.debug("No champion alias found for model", model_name=model_name)
            else:
                raise

        # 2. & 3. Select by stage and timestamp
        versions: list[ModelVersion] = self.client.get_latest_versions(model_name)
        if not versions:
            logger.warning("No versions found for model", model_name=model_name)
            return None

        # Sort versions by timestamp (most recent first)
        versions.sort(key=attrgetter("last_updated_timestamp"), reverse=True)

        for stage in self.stage_preference:
            stage_versions = [
                v for v in versions if ModelStage.from_str(v.current_stage) == stage
            ]
            if stage_versions:
                if len(stage_versions) > 1:
                    logger.info(
                        "Multiple versions found in stage, selecting latest by timestamp",
                        model_name=model_name,
                        stage=stage.value,
                        num_versions=len(stage_versions),
                    )
                selected_version = stage_versions[0]  # Already sorted by timestamp
                logger.info(
                    "Selected model version by stage and timestamp",
                    model_name=model_name,
                    stage=stage.value,
                    version=selected_version.version,
                )
                return ModelInfo.model_validate(
                    self._model_version_to_dict(selected_version)
                )

        logger.warning(
            "Could not select a model version based on any rule", model_name=model_name
        )
        return None
