"""Tasks del Expert Academic Writer."""

from __future__ import annotations

from crewai import Agent, Task

from mk_paper.models.writing_brief import WritingBrief


def create_writer_task(agent: Agent, brief: WritingBrief) -> Task:
    """Crea la task de Intro + Revisión de Literatura alimentada por WritingBrief."""
    brief_json = brief.model_dump_json(indent=2)
    return Task(
        description=(
            "Redacta un manuscrito de revisión de literatura de nivel Q1-Q2 con "
            "monolinguismo absoluto en español.\n\n"
            "INPUT — WritingBrief (pásalo íntegro como writing_brief_json):\n"
            f"{brief_json}\n\n"
            "Workflow obligatorio:\n"
            "1) `Load Writing Inputs`.\n"
            "2) `Build Citation Catalog`.\n"
            "3) `Draft Literature Paper`.\n"
            "4) `Validate APA Citations` y corregir si hay cite_keys inválidos.\n"
            "5) Opcional: `Export Paper Formats`.\n\n"
            "Reglas duras de formato:\n"
            "- Idioma 100% español.\n"
            "- Citas solo como [@cite_key] o [@a; @b].\n"
            "- Prohibido JSON crudo, código o metadatos internos.\n"
            "- Prohibidas viñetas estructurales.\n"
            "- Estructura fija: Introducción (cierra con pregunta de investigación), "
            "Revisión de Literatura / Marco Teórico (de lo general a lo específico), "
            "Referencias APA.\n"
            "- Devolver `PaperDraft` JSON final.\n"
        ),
        expected_output=(
            "JSON PaperDraft con markdown Intro + Revisión de Literatura en español "
            "puro, pregunta de investigación al cierre de la Intro, sin viñetas, "
            "con citas Pandoc válidas y Referencias APA."
        ),
        agent=agent,
    )
