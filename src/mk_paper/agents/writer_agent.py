"""Factory del Agente Redactor Experto (Expert Academic Writer)."""

from __future__ import annotations

from crewai import Agent, LLM

from mk_paper.config.llm import get_llm
from mk_paper.tools.writer_tools import (
    build_citation_catalog_tool,
    draft_literature_paper_tool,
    export_paper_formats_tool,
    load_writing_inputs_tool,
    validate_apa_citations_tool,
)


def create_scientific_writer(llm: LLM | str | None = None) -> Agent:
    """Crea el Expert Academic Writer con políticas editoriales Q1-Q2."""
    model = llm if llm is not None else get_llm()

    return Agent(
        role="Expert Academic Writer",
        goal=(
            "Redactar Introducción y Revisión de Literatura / Marco Teórico en "
            "español académico estricto, con narrativa de lo general a lo "
            "específico, cierre de la Intro con la pregunta de investigación, "
            "citas exclusivamente [@cite_key] y Referencias APA 7 deterministas."
        ),
        backstory=(
            "Eres editor académico jefe especializado en marcos teóricos. El texto "
            "final debe leerse como un artículo científico profesional, no como "
            "salida de software. Reglas inflexibles: idioma 100% español, prosa "
            "continua sin viñetas, cero JSON o metadatos internos, y citas solo "
            "con etiquetas Pandoc [@cite_key]. La Introducción ofrece panorama "
            "sin profundizar en exceso y termina con la pregunta de investigación; "
            "la Revisión de Literatura profundiza de lo general a lo específico "
            "usando literatura seminal, conceptual y core."
        ),
        tools=[
            load_writing_inputs_tool,
            build_citation_catalog_tool,
            draft_literature_paper_tool,
            validate_apa_citations_tool,
            export_paper_formats_tool,
        ],
        llm=model,
        verbose=True,
        allow_delegation=False,
        max_iter=10,
    )
