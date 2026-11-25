"""Central application configuration loaded from environment variables."""

from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime settings.

    Schema: environment variables are case-insensitive and map to these fields. Secrets must
    remain in the process environment or a secret manager; never place them in prompts/logs.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qwen_api_base: str = ""
    qwen_api_key: SecretStr = SecretStr("")
    qwen_model: str = "Qwen/Qwen3-8B"  # Track B guide: same model id for every group
    qwen_ca_cert: str = ""  # private CA for the Qwen endpoint; relative to project root
    qwen_timeout_seconds: int = Field(default=45, ge=1, le=600)
    # Shared capstone embedding API (OpenAI-compatible /v1/embeddings); see src/verisheet/embeddings.py.
    embedding_provider: str = "auto"  # auto | api | local
    embedding_api_base: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "bge-small-en-v1.5"
    embedding_ca_cert: str = ""
    embedding_timeout_seconds: int = Field(default=30, ge=1, le=300)
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    trace_dir: str = "results/traces"
    max_upload_mb: int = Field(default=50, ge=1, le=500)
    max_sample_rows: int = Field(default=50, ge=1, le=500)
    max_agent_actions: int = Field(default=10, ge=1, le=50)
    max_repairs: int = Field(default=2, ge=0, le=5)
    execution_timeout_seconds: int = Field(default=20, ge=1, le=120)
    execution_memory_mb: int = Field(default=1024, ge=128, le=8192)
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings for the current process.

    Returns:
        Settings: validated application configuration.
    """
    return Settings()
