"""Configuración centralizada del proyecto."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _empty_str_to_none(value: object) -> object:
    """Convierte strings vacíos o solo espacios en None (secrets opcionales)."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


BlankOptionalStr = Annotated[str | None, BeforeValidator(_empty_str_to_none)]

_DOCKER_PATH_PREFIX = "/app/"
_LOCAL_PATH_MAP: dict[str, str] = {
    "data_dir": "data",
    "output_dir": "output",
    "workspace_dir": "data/workspace",
}


def _find_project_root() -> Path | None:
    """Localiza la raíz del repo (directorio con ``pyproject.toml``)."""
    candidates: list[Path] = [Path.cwd().resolve()]
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        if (base / "pyproject.toml").is_file():
            return base
    return None


def _resolve_docker_path_to_local(path_value: str, *, local_relative: str) -> str:
    """Mapea rutas Docker ``/app/...`` a rutas locales cuando no hay contenedor."""
    raw = (path_value or "").strip()
    path = Path(raw)
    if not raw.startswith(_DOCKER_PATH_PREFIX):
        return raw

    root = _find_project_root()
    if root is None:
        return raw

    local = (root / local_relative).resolve()
    if path.exists():
        try:
            probe = path / ".mk_paper_write_probe"
            probe.touch()
            probe.unlink()
            return raw
        except OSError:
            return str(local)

    local.mkdir(parents=True, exist_ok=True)
    return str(local)


class Settings(BaseSettings):
    """Variables de entorno y configuración de la aplicación."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM — Groq es el proveedor principal (velocidad / bajo costo)
    openai_api_key: BlankOptionalStr = None
    groq_api_key: BlankOptionalStr = None
    deepseek_api_key: BlankOptionalStr = None
    openrouter_api_key: BlankOptionalStr = None
    litellm_model: str = "groq/llama-3.3-70b-versatile"
    litellm_fast_model: str = "groq/llama-3.1-8b-instant"
    llm_temperature: float = 0.2

    # AWS / S3
    aws_access_key_id: BlankOptionalStr = None
    aws_secret_access_key: BlankOptionalStr = None
    aws_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    s3_bucket: str = "mk-paper"

    # Database
    database_url: str = "postgresql://mkpaper:mkpaper@postgres:5432/mkpaper"

    # Paths
    workspace_dir: str = "/app/workspace"
    output_dir: str = "/app/output"
    data_dir: str = "/app/data"

    # Literature APIs
    unpaywall_email: BlankOptionalStr = None
    semantic_scholar_api_key: BlankOptionalStr = Field(
        default=None,
        description=(
            "API key de Semantic Scholar Graph API. Opcional: sin ella la búsqueda "
            "funciona con rate limits más bajos. Header: x-api-key."
        ),
    )
    http_timeout_seconds: float = 45.0
    http_max_retries: int = 4

    # PDF full-text extraction (OA papers)
    pdf_fulltext_enabled: bool = True
    pdf_max_pages: int = 12
    pdf_max_chars: int = 14000
    pdf_download_concurrency: int = 2

    # Hybrid alignment (TF-IDF + cosine)
    alignment_high_threshold: float = 0.35
    alignment_low_threshold: float = 0.12
    alignment_max_features: int = 12000

    # Seminal / foundational literature
    seminal_min_age_years: int = 10
    seminal_cites_per_year: float = 50.0
    seminal_min_citations: int = 500
    seminal_alignment_floor: float = 0.04
    seminal_history_boost: float = 0.15

    # App
    log_level: str = "INFO"


def _resolve_runtime_paths(settings: Settings) -> Settings:
    """En WSL/local, reemplaza ``/app/*`` por rutas del repo si no hay Docker."""
    updates: dict[str, str] = {}
    for field, rel in _LOCAL_PATH_MAP.items():
        current = str(getattr(settings, field))
        resolved = _resolve_docker_path_to_local(current, local_relative=rel)
        if resolved != current:
            updates[field] = resolved
    if updates:
        return settings.model_copy(update=updates)
    return settings


def get_settings() -> Settings:
    """Retorna una instancia de configuración con rutas resueltas al entorno."""
    return _resolve_runtime_paths(Settings())
