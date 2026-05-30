from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    app_url: str = Field(default="http://localhost:8000")
    secret_key: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=60)
    rate_limit_per_minute: int = Field(default=100)

    # ── OpenRouter ────────────────────────────────────────────────────────────
    openrouter_api_key: str = Field(default="")
    openrouter_model_heavy: str = Field(default="anthropic/claude-3.5-sonnet")
    openrouter_model_light: str = Field(default="anthropic/claude-3-haiku")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/mednlp"
    )
    database_url_sync: str = Field(default="postgresql://postgres:postgres@localhost:5432/mednlp")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_api_key: str = Field(default="")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")

    # ── GCP ───────────────────────────────────────────────────────────────────
    gcp_project_id: str = Field(default="mednlp-dev")
    gcp_region: str = Field(default="us-central1")
    pubsub_topic_notes: str = Field(default="notes.incoming")
    pubsub_subscription_notes: str = Field(default="notes.processor")
    pubsub_topic_dlq: str = Field(default="notes.dlq")
    gcs_bucket_raw: str = Field(default="mednlp-raw-notes-dev")
    gcs_bucket_processed: str = Field(default="mednlp-processed-notes-dev")
    gcs_bucket_artifacts: str = Field(default="mednlp-model-artifacts-dev")
    pubsub_emulator_host: str = Field(default="")

    # ── Alerting ──────────────────────────────────────────────────────────────
    pagerduty_integration_key: str = Field(default="")
    slack_webhook_url: str = Field(default="")

    # ── Drift thresholds ──────────────────────────────────────────────────────
    embedding_drift_threshold: float = Field(default=0.1)
    dlq_alert_rate_threshold: float = Field(default=0.05)

    @model_validator(mode="after")
    def warn_insecure_defaults(self) -> "Settings":
        if self.env == "production" and self.secret_key == "change-me-in-production":  # noqa: S105
            raise ValueError("SECRET_KEY must be set in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def pubsub_emulator_enabled(self) -> bool:
        return bool(self.pubsub_emulator_host)


@lru_cache
def get_settings() -> Settings:
    return Settings()
