"""Factory del Agente Investigador (Literature Reviewer)."""

from __future__ import annotations

from crewai import Agent, LLM

from mk_paper.config.settings import Settings, get_settings
from mk_paper.tools.literature_tools import search_scientific_literature


def _resolve_llm(llm: str | None, settings: Settings) -> str | LLM:
    """Resuelve el modelo LiteLLM según override o API keys disponibles."""
    if llm:
        return llm

    if settings.openrouter_api_key:
        return LLM(
            model=settings.litellm_fast_model,
            api_key=settings.openrouter_api_key,
        )
    if settings.groq_api_key:
        return LLM(
            model="groq/llama-3.1-8b-instant",
            api_key=settings.groq_api_key,
        )
    if settings.openai_api_key:
        return LLM(model=settings.litellm_model, api_key=settings.openai_api_key)
    if settings.deepseek_api_key:
        return LLM(
            model="deepseek/deepseek-chat",
            api_key=settings.deepseek_api_key,
        )

    # Sin keys: devolver string configurado; CrewAI fallará al instanciar si
    # el provider exige API key. El caller puede pasar llm= explícitamente.
    return settings.litellm_fast_model


def create_literature_reviewer(llm: str | None = None) -> Agent:
    """Crea el agente Literature Reviewer con la tool de búsqueda bibliográfica.

    Args:
        llm: Identificador LiteLLM opcional. Si es None, usa
            ``settings.litellm_fast_model`` cuando hay API key compatible,
            o cae a otros providers configurados.

    Returns:
        Agente CrewAI configurado para localizar literatura Q1/Q2 con DOI y OA.
    """
    settings = get_settings()
    model = _resolve_llm(llm, settings)

    return Agent(
        role="Literature Reviewer",
        goal=(
            "Localizar literatura científica reciente de alto impacto (Q1/Q2), "
            "filtrar por calidad metodológica y acceso abierto, y devolver "
            "evidencia estructurada con DOI, metadatos y enlaces a texto completo."
        ),
        backstory=(
            "Eres un investigador académico experto en revisión sistemática de "
            "literatura. Priorizas papers con DOI válido y, cuando existe, acceso "
            "abierto al PDF. Evitas sesgo de confirmación: reportas hallazgos "
            "contradictorios y limitaciones. Usas la herramienta de búsqueda "
            "científica para combinar Semantic Scholar con OpenAlex/Unpaywall y "
            "trabajas solo con evidencia verificable."
        ),
        tools=[search_scientific_literature],
        llm=model,
        verbose=True,
        allow_delegation=False,
    )
