import pytest

from src.core.config import Settings


class TestSettings:
    def test_defaults_load_without_env_file(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.env == "development"
        assert settings.openrouter_model_heavy == "anthropic/claude-3.5-sonnet"

    def test_production_blocks_insecure_secret_key(self) -> None:
        with pytest.raises(ValueError, match="SECRET_KEY must be set"):
            Settings(env="production", secret_key="change-me-in-production")

    def test_pubsub_emulator_enabled_when_host_set(self) -> None:
        settings = Settings(pubsub_emulator_host="localhost:8085")
        assert settings.pubsub_emulator_enabled is True

    def test_pubsub_emulator_disabled_when_host_empty(self) -> None:
        settings = Settings(pubsub_emulator_host="")
        assert settings.pubsub_emulator_enabled is False

    def test_is_production_flag(self) -> None:
        dev = Settings(env="development")
        prod = Settings(env="production", secret_key="a-secure-key-123456789012")
        assert dev.is_production is False
        assert prod.is_production is True
