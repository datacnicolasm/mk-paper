"""Configuración centralizada del proyecto."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Variables de entorno y configuración de la aplicación."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    deepseek_api_key: str | None = None
    openrouter_api_key: str | None = None
    litellm_model: str = "gpt-4o"
    litellm_fast_model: str = "openrouter/meta-llama/llama-3.1-8b-instruct"

    # AWS / S3
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    s3_bucket: str = "mk-paper"

    # Database
    database_url: str = "postgresql://mkpaper:mkpaper@postgres:5432/mkpaper"

    # Paths
    workspace_dir: str = "/app/workspace"
    output_dir: str = "/app/output"
    data_dir: str = "/app/data"

    # App
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Retorna una instancia de configuración."""
    return Settings()
