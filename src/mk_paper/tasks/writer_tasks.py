"""Tasks del Scientific Writer."""

from __future__ import annotations

from crewai import Agent, Task

from mk_paper.models.writing_brief import WritingBrief


def create_writer_task(agent: Agent, brief: WritingBrief) -> Task:
    """Crea la task IMRaD alimentada por un WritingBrief.

    Args:
        agent: Scientific Writer configurado.
        brief: Brief con paths a literatura + análisis.

    Returns:
        Task CrewAI que exige PaperDraft JSON validado.
    """
    brief_json = brief.model_dump_json(indent=2)
    return Task(
        description=(
            "Redacta un artículo científico IMRaD fusionando literatura local y "
            "el reporte cuantitativo. El núcleo de citas es Pandoc [@cite_key].\n\n"
            "INPUT — WritingBrief (pásalo íntegro como writing_brief_json):\n"
            f"{brief_json}\n\n"
            "Workflow obligatorio:\n"
            "1) `Load Writing Inputs`.\n"
            "2) `Build Citation Catalog` — memoriza cite_keys; prohibido citar "
            "fuera del catálogo.\n"
            "3) `Draft IMRAD Paper`.\n"
            "4) `Validate APA Citations` (valida solo patrones [@key]); si hay "
            "citations_unknown, corrige y reintenta.\n"
            "5) Opcional: `Export Paper Formats`.\n\n"
            "Reglas duras:\n"
            "- Citas SOLO como [@beneish1999] o [@beneish1999; @piotroski2000]. "
            "Nunca escribas 'Autor (Año)' a mano.\n"
            "- Introducción/marco teórico: seminal + conceptual.\n"
            "- Discusión: core para benchmarking empírico.\n"
            "- Resultados: no reescribir tablas; el sistema inyecta tablas "
            "literales del AnalysisReport.\n"
            "- Sin inventar DOIs/autores/métricas.\n"
            "- Estructura: Introducción, Metodología, Resultados, Discusión, "
            "Referencias (APA determinista).\n"
            "- Devuelve PaperDraft JSON final.\n"
        ),
        expected_output=(
            "JSON PaperDraft con title, markdown (IMRaD + Referencias APA), "
            "latex opcional, citations_used (cite_keys), validation, warnings, status."
        ),
        agent=agent,
    )
