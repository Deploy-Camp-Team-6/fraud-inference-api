from typing import Any, Optional, cast

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
    API_KEYS: list[SecretStr] = ["dev-key-1", "dev-key-2"]

    # Model & Prediction Configuration
    SERVICE_THRESHOLD: float = 0.5
    ALLOW_EXTRA: bool = False

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
    def assemble_api_keys(cls, v: Any) -> list[SecretStr]:
        if isinstance(v, str):
            return [SecretStr(key.strip()) for key in v.split(",") if key.strip()]
        return cast(list[SecretStr], v)

settings = Settings()
