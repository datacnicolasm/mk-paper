"""Factory del Agente Analista Cuantitativo (Quantitative Analyst)."""

from __future__ import annotations

from crewai import Agent, LLM

from mk_paper.config.llm import get_llm
from mk_paper.tools.analysis_tools import (
    execute_python_analysis_tool,
    run_quantitative_analysis_tool,
    validate_and_preprocess_dataset_tool,
)


def create_quantitative_analyst(llm: LLM | str | None = None) -> Agent:
    """Crea el Quantitative Analyst con REPL dinámico + auto-corrección.

    El agente escribe Python a la medida del MethodBrief y lo ejecuta sobre el
    DataFrame local vía ``Execute Python Analysis``. Ante errores lee el
    traceback, corrige el script y reintenta.

    Args:
        llm: Instancia ``crewai.LLM``, string LiteLLM, o None para Groq.

    Returns:
        Agente CrewAI con REPL científico y tools auxiliares.
    """
    model = llm if llm is not None else get_llm()

    return Agent(
        role="Quantitative Analyst",
        goal=(
            "Traducir el MethodBrief del investigador en código Python científico "
            "dinámico, ejecutarlo sobre el CSV/XLSX local con Execute Python "
            "Analysis (pandas/numpy/sklearn/statsmodels/scipy/xgboost), obtener "
            "métricas/tablas en `results`, y si hay error usar el traceback para "
            "auto-corregir y reejecutar hasta éxito."
        ),
        backstory=(
            "Eres un econometra/científico de datos Q1-Q2 con máxima flexibilidad "
            "metodológica. Puedes implementar PCA, Random Forest, regresiones, "
            "XGBoost u otras técnicas pedidas en el brief escribiendo Python. "
            "No buscas datos en internet: solo el dataset local. "
            "Workflow: (1) opcional Validate And Preprocess Dataset; "
            "(2) escribe código que usa `df` y asigna un dict a `results`; "
            "(3) llama Execute Python Analysis; (4) si status=error, lee "
            "traceback/message, corrige el script y reintenta (self-correction). "
            "Run Quantitative Analysis sigue disponible como atajo determinista "
            "cuando el brief encaje en wrappers whitelist. No inventes cifras."
        ),
        tools=[
            validate_and_preprocess_dataset_tool,
            execute_python_analysis_tool,
            run_quantitative_analysis_tool,
        ],
        llm=model,
        verbose=True,
        allow_delegation=False,
        max_iter=12,
    )
