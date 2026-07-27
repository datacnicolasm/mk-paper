"""Tasks del Quantitative Analyst."""

from __future__ import annotations

from crewai import Agent, Task

from mk_paper.models.method_brief import MethodBrief


def create_analysis_task(agent: Agent, brief: MethodBrief) -> Task:
    """Crea la task de análisis cuantitativo alimentada por un MethodBrief.

    Args:
        agent: Quantitative Analyst configurado.
        brief: Brief metodológico estructurado (puede incluir literatura local).

    Returns:
        Task CrewAI que exige resultados numéricos vía REPL dinámico.
    """
    brief_json = brief.model_dump_json(indent=2)
    return Task(
        description=(
            "Ejecuta el análisis cuantitativo pedido en el MethodBrief escribiendo "
            "y ejecutando Python dinámico sobre el dataset local.\n\n"
            "INPUT — MethodBrief:\n"
            f"{brief_json}\n\n"
            "Workflow (flexibilidad metodológica):\n"
            "1) Opcional: `Validate And Preprocess Dataset` con el MethodBrief JSON.\n"
            "2) Escribe un script Python a la medida (PCA, RF, OLS, XGBoost, etc.). "
            "El entorno ya tiene `df` (DataFrame cargado), `pd`, `np`, `sklearn`, "
            "`scipy`, `sm`/`statsmodels`, `xgb`/`xgboost`. Asigna un dict a "
            "`results` con métricas/tablas (p.ej. best_model, metrics, coefficients).\n"
            "3) Llama `Execute Python Analysis` con dataset_path del brief y "
            "python_code. sheet_name si aplica.\n"
            "4) SELF-CORRECTION: si status=error, lee traceback/message/stdout, "
            "corrige el script y reejecuta hasta status=ok. No abandones ante "
            "errores de sintaxis, dimensiones o tipos.\n"
            "5) Atajo: si el brief encaja en wrappers whitelist, puedes usar "
            "`Run Quantitative Analysis` en su lugar.\n"
            "6) No busques datos en internet. No inventes métricas ni DOIs.\n"
            "7) Entrega un resumen JSON final con results exitosos + warnings.\n"
        ),
        expected_output=(
            "JSON con status=ok, dataset_path, results (métricas/tablas del "
            "script), stdout relevante y notes metodológicas. Si usaste el "
            "atajo determinista, AnalysisReport completo."
        ),
        agent=agent,
    )
