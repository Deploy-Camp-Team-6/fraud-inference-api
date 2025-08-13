import json
from typing import Any, Optional, cast, Literal

from pydantic import field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.
    """

    # Service Info
    BUILD_TIME: str = "dev"
    GIT_SHA: str = "dev"

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = ""
    MLFLOW_TRACKING_USERNAME: Optional[str] = None
    MLFLOW_TRACKING_PASSWORD: Optional[SecretStr] = None
    MLFLOW_S3_ENDPOINT_URL: Optional[str] = None

    # AWS Configuration (if using S3)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[SecretStr] = None
    AWS_DEFAULT_REGION: Optional[str] = "us-east-1"

    # API Security
    API_KEYS: list[str] = ["dev-key-1", "dev-key-2"]

    # Model & Prediction Configuration
    SERVICE_THRESHOLD: float = 0.5
    ALLOW_EXTRA: bool = False

    # Dependency Validation Rules
    DEPENDENCY_VALIDATION_RULES: dict[str, Literal["exact", "compatible"]] = {
        "scikit-learn": "compatible",
        "xgboost": "compatible",
        "lightgbm": "compatible",
        "pandas": "compatible",
        "numpy": "compatible",
        "mlflow": "compatible",
    }

    # Web Server Configuration
    LOG_LEVEL: str = "info"
    WORKERS: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("API_KEYS", mode="before")
    @classmethod
    def assemble_api_keys(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return cast(list[str], v)

    @field_validator("DEPENDENCY_VALIDATION_RULES", mode="before")
    @classmethod
    def assemble_validation_rules(
        cls, v: Any
    ) -> dict[str, Literal["exact", "compatible"]]:
        if isinstance(v, str):
            return json.loads(v)
        return cast(dict[str, Literal["exact", "compatible"]], v)


settings = Settings()
