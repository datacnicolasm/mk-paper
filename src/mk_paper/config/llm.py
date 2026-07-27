"""Instancia central de LLM para agentes CrewAI (via LiteLLM / Groq)."""

from __future__ import annotations

import logging
import os

from crewai import LLM

from mk_paper.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_DEFAULT_GROQ_MODEL = "groq/llama-3.3-70b-versatile"
_DEFAULT_GROQ_FAST_MODEL = "groq/llama-3.1-8b-instant"


def _ensure_groq_model(model: str) -> str:
    """Garantiza el prefijo LiteLLM ``groq/`` para no caer en otro provider."""
    cleaned = (model or "").strip()
    if not cleaned:
        return _DEFAULT_GROQ_MODEL
    if cleaned.startswith("groq/"):
        return cleaned
    logger.warning(
        "Modelo %r no usa prefijo groq/; se fuerza a groq/%s",
        cleaned,
        cleaned,
    )
    return f"groq/{cleaned}"


def get_llm(
    *,
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> LLM:
    """Crea el LLM principal del proyecto **exclusivamente con Groq**.

    Usa la integración LiteLLM de CrewAI. Requiere ``GROQ_API_KEY`` en el
    entorno / ``.env``. No hay fallback a OpenRouter ni OpenAI.

    Args:
        model: Override del modelo LiteLLM (ej. ``groq/llama-3.3-70b-versatile``).
        temperature: Temperatura de muestreo. Si es None, usa settings.
        settings: Settings inyectables (tests). Si es None, lee el entorno.

    Returns:
        Instancia ``crewai.LLM`` lista para pasar a ``Agent(llm=...)``.

    Raises:
        ValueError: Si ``GROQ_API_KEY`` no está configurada.
    """
    cfg = settings or get_settings()
    api_key = cfg.groq_api_key
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY no está configurada. Agrégala en el archivo .env "
            "(ver .env.example). mk-paper usa Groq como único provider de LLM."
        )

    resolved_model = _ensure_groq_model(model or cfg.litellm_model)
    resolved_temp = cfg.llm_temperature if temperature is None else temperature

    # Forzar en el proceso para LiteLLM/CrewAI (sobrescribe valores vacíos previos).
    os.environ["GROQ_API_KEY"] = api_key
    # Evitar que un OPENROUTER_API_KEY residual desvíe el routing.
    if not cfg.openrouter_api_key:
        os.environ.pop("OPENROUTER_API_KEY", None)

    logger.info("LLM provider=groq model=%s", resolved_model)
    return LLM(
        model=resolved_model,
        api_key=api_key,
        temperature=resolved_temp,
    )


def get_fast_llm(
    *,
    settings: Settings | None = None,
) -> LLM:
    """LLM económico/rápido en Groq para tareas masivas."""
    cfg = settings or get_settings()
    fast = _ensure_groq_model(cfg.litellm_fast_model or _DEFAULT_GROQ_FAST_MODEL)
    return get_llm(model=fast, settings=cfg)
