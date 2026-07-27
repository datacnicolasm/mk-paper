"""Factory del Agente Investigador (Literature Reviewer)."""

from __future__ import annotations

from crewai import Agent, LLM

from mk_paper.config.llm import get_llm
from mk_paper.tools.literature_tools import search_scientific_literature


def create_literature_reviewer(llm: LLM | str | None = None) -> Agent:
    """Crea el agente Literature Reviewer con la tool de búsqueda bibliográfica.

    Args:
        llm: Instancia ``crewai.LLM``, string LiteLLM, o None para usar el
            LLM central de Groq (``get_llm()``).

    Returns:
        Agente CrewAI configurado para localizar literatura Q1/Q2 con DOI y OA.
    """
    model = llm if llm is not None else get_llm()

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
