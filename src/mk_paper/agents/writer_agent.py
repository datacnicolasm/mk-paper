"""Factory del Agente Redactor Científico (Scientific Writer)."""

from __future__ import annotations

from crewai import Agent, LLM

from mk_paper.config.llm import get_llm
from mk_paper.tools.writer_tools import (
    build_citation_catalog_tool,
    draft_imrad_paper_tool,
    export_paper_formats_tool,
    load_writing_inputs_tool,
    validate_apa_citations_tool,
)


def create_scientific_writer(llm: LLM | str | None = None) -> Agent:
    """Crea el Scientific Writer (IMRaD + APA 7 vía cite_keys Pandoc).

    Args:
        llm: Instancia ``crewai.LLM``, string LiteLLM, o None para Groq.

    Returns:
        Agente CrewAI con tools de carga, catálogo APA, draft, validación y export.
    """
    model = llm if llm is not None else get_llm()

    return Agent(
        role="Scientific Writer",
        goal=(
            "Fusionar el JSON del Literature Reviewer y el AnalysisReport en un "
            "artículo IMRaD Q1-Q2. Citar SOLO con cite_keys Pandoc [@key]; el "
            "sistema expande a APA 7 y genera Referencias. No inventar fuentes "
            "ni reescribir tablas numéricas."
        ),
        backstory=(
            "Eres editor académico Q1-Q2. Tono objetivo y mesurado; sin marketing. "
            "Protocolo de citas: obligatoriamente [@cite_key] o "
            "[@key1; @key2] — nunca APA autor-año a mano. "
            "Introducción/marco teórico ≈ seminal+conceptual; Discusión ≈ core "
            "para benchmarking empírico. En Resultados insertas placeholders de "
            "tablas literales del skeleton; no redondeas ni retipeas métricas. "
            "Workflow: Load Writing Inputs → Build Citation Catalog → "
            "Draft IMRAD Paper → Validate APA Citations (solo Pandoc keys) → "
            "Export Paper Formats."
        ),
        tools=[
            load_writing_inputs_tool,
            build_citation_catalog_tool,
            draft_imrad_paper_tool,
            validate_apa_citations_tool,
            export_paper_formats_tool,
        ],
        llm=model,
        verbose=True,
        allow_delegation=False,
        max_iter=10,
    )
