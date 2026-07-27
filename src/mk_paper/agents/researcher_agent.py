"""Factory del Agente Investigador (Literature Reviewer)."""

from __future__ import annotations

from crewai import Agent, LLM

from mk_paper.config.llm import get_llm
from mk_paper.tools.systematic_review import run_systematic_literature_review


def create_literature_reviewer(llm: LLM | str | None = None) -> Agent:
    """Crea el agente Literature Reviewer del embudo sistemático multinivel.

    Args:
        llm: Instancia ``crewai.LLM``, string LiteLLM, o None para usar Groq
            vía ``get_llm()``.

    Returns:
        Agente CrewAI que ejecuta la tool de revisión sistemática y produce
        JSON Writer-ready (Core Findings + Conceptual References).
    """
    model = llm if llm is not None else get_llm()

    return Agent(
        role="Literature Reviewer",
        goal=(
            "Ejecutar una revisión sistemática rigurosa a partir de un ResearchBrief: "
            "matriz de búsqueda, recuperación dual (S2 + OpenAlex/Unpaywall), filtro "
            "híbrido TF-IDF/cosine (alignment_score), centralidad histórica y "
            "whitelist seminal; clasificación Nivel 1 Core / Nivel 2 Conceptual / "
            "Seminal Literature, con evidencia JSON para el Academic Writer."
        ),
        backstory=(
            "Eres un especialista en revisiones sistemáticas Q1/Q2. No dependes solo "
            "del juicio del LLM: vectorizas el perfil de investigación y los papers "
            "(título+abstract+keywords) con TF-IDF, calculas similitud de coseno y "
            "aplicas umbrales duros (alto→Core auto, medio→Groq conceptual, bajo→"
            "descarte). Preservas literatura fundacional vía whitelist de DOIs y "
            "centralidad histórica (citas/años) en una sección Seminal separada del "
            "Nivel 1 empírico. Reportas alignment_score numérico en cada cita."
        ),
        tools=[run_systematic_literature_review],
        llm=model,
        verbose=True,
        allow_delegation=False,
    )
