"""Paquete de configuración (settings + LLM central)."""

from mk_paper.config.llm import get_fast_llm, get_llm
from mk_paper.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "get_llm", "get_fast_llm"]
