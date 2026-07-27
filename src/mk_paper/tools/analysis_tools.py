"""Herramientas de análisis cuantitativo determinista (solo datos locales).

Sin acceso a internet. El LLM no inventa metodología ni ejecuta código libre:
recibe un MethodBrief y un CSV/XLSX local, y un motor fijo evalúa modelos
whitelist + grillas de hiperparámetros acotadas.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from crewai.tools import tool
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from mk_paper.config.llm import get_llm
from mk_paper.config.settings import Settings, get_settings
from mk_paper.models.method_brief import (
    ALLOWED_CLASSIFICATION_MODELS,
    ALLOWED_REGRESSION_MODELS,
    PRIMARY_METRIC_DIRECTION,
    AnalysisReport,
    LiteratureBenchmark,
    MethodBrief,
    ModelResult,
    DatasetSchema,
)
from mk_paper.tools.model_wrappers import (
    dispatch_supervised_wrapper,
    run_pca,
)
from mk_paper.tools.safe_python_repl import execute_analysis_code

logger = logging.getLogger(__name__)


def resolve_sandbox_path(
    raw_path: str,
    *,
    settings: Settings | None = None,
    must_exist: bool = True,
) -> Path:
    """Resuelve path bajo data/workspace/output (misma sandbox que datasets)."""
    cfg = settings or get_settings()
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        for base in (Path(cfg.data_dir), Path(cfg.workspace_dir), Path(cfg.output_dir), Path.cwd()):
            trial = (base / candidate).resolve()
            if trial.exists() or not must_exist:
                candidate = trial
                if trial.exists():
                    break
        else:
            candidate = (Path(cfg.data_dir) / raw_path).resolve()
    else:
        candidate = candidate.resolve()

    allowed_roots = [
        Path(cfg.data_dir).resolve(),
        Path(cfg.workspace_dir).resolve(),
        Path(cfg.output_dir).resolve(),
    ]
    if not any(_is_relative_to(candidate, root) for root in allowed_roots):
        raise PermissionError(
            f"Path outside sandbox (data/workspace/output): {candidate}"
        )
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"File not found: {candidate}")
    return candidate


def resolve_dataset_path(
    raw_path: str,
    *,
    settings: Settings | None = None,
) -> Path:
    """Resuelve y valida que el path esté dentro de data/workspace/output."""
    candidate = resolve_sandbox_path(raw_path, settings=settings, must_exist=True)
    if candidate.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        raise ValueError(f"Unsupported dataset extension: {candidate.suffix}")
    return candidate


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _paper_card(paper: dict[str, Any], *, level: str) -> dict[str, Any]:
    """Compacta un paper del Literature Reviewer a evidencia usable."""
    findings = paper.get("key_findings") or []
    if not isinstance(findings, list):
        findings = [str(findings)]
    return {
        "level": level,
        "doi": paper.get("doi"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "utility": paper.get("utility"),
        "alignment_score": paper.get("alignment_score"),
        "key_findings": [str(f) for f in findings][:6],
        "citation_context": str(paper.get("citation_context") or "")[:500],
        "suggested_section": paper.get("suggested_section"),
    }


def compact_literature_review(payload: dict[str, Any] | str) -> dict[str, Any]:
    """Reduce review.json / markdown contextual a un snapshot estructurado."""
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return {
                    "brief_title": None,
                    "source_format": "markdown",
                    "markdown_excerpt": text[:6000],
                    "evidence": [],
                    "n_core": 0,
                    "n_conceptual": 0,
                    "n_seminal": 0,
                }
        else:
            return {
                "brief_title": None,
                "source_format": "markdown",
                "markdown_excerpt": text[:6000],
                "evidence": [],
                "n_core": 0,
                "n_conceptual": 0,
                "n_seminal": 0,
            }

    assert isinstance(payload, dict)
    evidence: list[dict[str, Any]] = []
    for paper in payload.get("core_findings") or []:
        if isinstance(paper, dict):
            evidence.append(_paper_card(paper, level="core"))
    for paper in payload.get("conceptual_references") or []:
        if isinstance(paper, dict):
            evidence.append(_paper_card(paper, level="conceptual"))
    for paper in payload.get("seminal_literature") or []:
        if isinstance(paper, dict):
            evidence.append(_paper_card(paper, level="seminal"))

    return {
        "brief_title": payload.get("brief_title"),
        "source_format": "json",
        "candidate_count": payload.get("candidate_count"),
        "discarded_count": payload.get("discarded_count"),
        "n_core": len(payload.get("core_findings") or []),
        "n_conceptual": len(payload.get("conceptual_references") or []),
        "n_seminal": len(payload.get("seminal_literature") or []),
        "evidence": evidence[:40],
        "warnings": list(payload.get("warnings") or [])[:10],
    }


def load_literature_context(
    brief: MethodBrief,
    *,
    settings: Settings | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Carga literatura local (path y/o inline). Sin red."""
    warnings: list[str] = []
    merged: dict[str, Any] = {
        "brief_title": None,
        "source_format": "none",
        "evidence": [],
        "n_core": 0,
        "n_conceptual": 0,
        "n_seminal": 0,
    }

    if brief.literature_context is not None:
        merged = compact_literature_review(brief.literature_context)
        warnings.append("Literature context loaded from MethodBrief.literature_context.")

    if brief.literature_review_path:
        try:
            path = resolve_sandbox_path(
                brief.literature_review_path, settings=settings, must_exist=True
            )
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                data = json.loads(text)
                file_snap = compact_literature_review(data)
            else:
                file_snap = compact_literature_review(text)
            # Path wins over empty inline; merge evidence if both present.
            if merged.get("evidence"):
                seen = {
                    (e.get("doi"), e.get("title")) for e in file_snap.get("evidence") or []
                }
                for card in merged.get("evidence") or []:
                    key = (card.get("doi"), card.get("title"))
                    if key not in seen:
                        file_snap.setdefault("evidence", []).append(card)
            merged = file_snap
            merged["source_path"] = str(path)
            warnings.append(f"Literature review loaded from {path}.")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not load literature_review_path: {exc}")

    return merged, warnings


def _benchmark_higher_is_better(bench: LiteratureBenchmark) -> bool:
    if bench.higher_is_better is not None:
        return bool(bench.higher_is_better)
    metric = bench.metric.strip().lower()
    direction = PRIMARY_METRIC_DIRECTION.get(metric)
    if direction == "maximize":
        return True
    if direction == "minimize":
        return False
    return True


def compare_metric_to_benchmarks(
    *,
    metric: str,
    value: float,
    benchmarks: list[LiteratureBenchmark],
    model_id: str | None = None,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Contrasta una métrica observada vs umbrales de literatura."""
    rows: list[dict[str, Any]] = []
    metric_l = metric.strip().lower()
    for bench in benchmarks:
        if bench.metric.strip().lower() != metric_l:
            continue
        hib = _benchmark_higher_is_better(bench)
        ref = float(bench.reference_value)
        delta = float(value) - ref
        if hib:
            verdict = "better" if value > ref else ("equal" if value == ref else "worse")
        else:
            verdict = "better" if value < ref else ("equal" if value == ref else "worse")
        rows.append(
            {
                "model_id": model_id,
                "params": params or {},
                "metric": metric_l,
                "observed": float(value),
                "literature_reference": ref,
                "delta_observed_minus_ref": delta,
                "higher_is_better": hib,
                "verdict": verdict,
                "source": bench.source,
                "note": bench.note,
            }
        )
    return rows


def _deterministic_discussion(
    brief: MethodBrief,
    report_bits: dict[str, Any],
    lit: dict[str, Any],
    benchmarking: list[dict[str, Any]],
) -> str:
    """Borrador estructurado para el Academic Writer (sin LLM)."""
    lines = [
        "## Analytical Discussion (for Academic Writer)",
        "",
        f"Study focus: **{brief.title}** (`{brief.task_type}`). "
        f"Best local model: `{report_bits.get('best_model')}` with "
        f"`{brief.primary_metric}` = {report_bits.get('best_score')}.",
        "",
        "### Relation to the state of the art",
    ]
    n_ev = len(lit.get("evidence") or [])
    if n_ev == 0 and not brief.literature_benchmarks:
        lines.append(
            "_No literature review context was supplied. Discuss numerical results "
            "as a self-contained empirical exercise and request a Literature Review "
            "JSON/Markdown for state-of-the-art alignment._"
        )
    else:
        lines.append(
            f"Literature snapshot: core={lit.get('n_core', 0)}, "
            f"conceptual={lit.get('n_conceptual', 0)}, "
            f"seminal={lit.get('n_seminal', 0)} "
            f"(title={lit.get('brief_title')!r})."
        )
        lines.append("")
        lines.append("Key evidence cards to cite:")
        for card in (lit.get("evidence") or [])[:8]:
            lines.append(
                f"- [{card.get('level')}] {card.get('title')} "
                f"({card.get('year')}; DOI `{card.get('doi')}`). "
                f"Use in: {card.get('suggested_section') or 'Discussion'}. "
                f"Context: {card.get('citation_context') or '—'}"
            )

    lines.extend(["", "### Benchmarking vs literature thresholds"])
    if not benchmarking:
        lines.append(
            "_No structured `literature_benchmarks` matched the primary metric. "
            "If the Literature Reviewer reported RMSE/R²/accuracy in key_findings, "
            "encode them as literature_benchmarks in the MethodBrief for quantitative contrast._"
        )
    else:
        # Prefer best-model rows
        best_rows = [
            r
            for r in benchmarking
            if r.get("model_id") == report_bits.get("best_model")
        ] or benchmarking
        for row in best_rows[:12]:
            lines.append(
                f"- Metric `{row['metric']}`: observed={row['observed']:.6g} vs "
                f"lit={row['literature_reference']:.6g} "
                f"(Δ={row['delta_observed_minus_ref']:.6g}) → **{row['verdict']}** "
                f"[{row.get('source') or 'literature'}]. {row.get('note') or ''}"
            )

    lines.extend(
        [
            "",
            "### Writer guidance",
            "- Report the model_comparison table in Results; do not invent extra metrics.",
            "- In Discussion, explicitly state whether results are better/worse/equal "
            "relative to each literature benchmark above.",
            "- Use seminal cards only for theoretical origins; use core/conceptual for "
            "empirical contrast.",
            "- Acknowledge dataset scope, sample size, and that no web search was used "
            "at the analysis stage.",
            "",
        ]
    )
    return "\n".join(lines)


def _llm_enrich_discussion(
    draft: str,
    *,
    brief: MethodBrief,
    lit: dict[str, Any],
    benchmarking: list[dict[str, Any]],
    report_bits: dict[str, Any],
) -> tuple[str, list[str]]:
    """Enriquece la discusión con Groq usando solo contexto local (sin web)."""
    warnings: list[str] = []
    prompt = f"""You are a Q1 quantitative researcher writing guidance for an Academic Writer.

You MUST NOT invent papers, DOIs, or numeric results. Use ONLY the provided materials.
Return Markdown (no JSON fences) refining the Analytical Discussion:
- Contrast our metrics with literature benchmarks and evidence cards.
- Tell the Writer where to place claims (Results vs Discussion vs Theoretical framework).
- Flag gaps when literature evidence is thin or off-topic.
- Keep a formal academic tone in English or Spanish matching the draft.

METHOD BRIEF TITLE: {brief.title}
TASK: {brief.task_type}
PRIMARY METRIC: {brief.primary_metric}
BEST MODEL / SCORE: {report_bits.get('best_model')} / {report_bits.get('best_score')}

LITERATURE SNAPSHOT (local, already curated):
{json.dumps(lit, ensure_ascii=False, indent=2)[:8000]}

BENCHMARKING ROWS:
{json.dumps(benchmarking[:30], ensure_ascii=False, indent=2)}

DRAFT TO REFINE:
{draft}
"""
    try:
        llm = get_llm()
        text = str(llm.call(prompt)).strip()
        if len(text) < 40:
            warnings.append("LLM discussion enrichment returned empty/short text; kept draft.")
            return draft, warnings
        return text, warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"LLM discussion enrichment skipped: {exc}")
        return draft, warnings


def load_local_dataset(
    path: str | Path,
    *,
    sheet_name: str | None = None,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Carga CSV/XLSX local (sin red)."""
    resolved = resolve_dataset_path(str(path), settings=settings)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(resolved)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved, sheet_name=sheet_name or 0, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Excel sheet did not resolve to a single DataFrame")
    return df, resolved


def validate_columns(
    df: pd.DataFrame,
    *,
    target: str,
    features: list[str],
) -> list[str]:
    """Verifica columnas requeridas; retorna lista de faltantes."""
    needed = [target, *features]
    return [c for c in needed if c not in df.columns]


def preprocess_analysis_frame(
    df: pd.DataFrame,
    *,
    target: str,
    predictors: list[str],
    impute_strategy: str = "drop",
    dropna: bool = True,
    task_type: str = "regression",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Preprocesado seguro: tipos numéricos, nulos, validación mínima.

    Returns:
        (x, y, work_df, warnings)
    """
    warnings: list[str] = []
    missing = validate_columns(df, target=target, features=predictors)
    if missing:
        available = [str(c) for c in df.columns]
        raise ValueError(
            "Columns not found in dataset: "
            f"{missing}. Available columns: {available}"
        )

    needed = [target, *predictors]
    work = df[needed].copy()
    non_numeric_notes: list[str] = []
    for col in needed:
        converted = pd.to_numeric(work[col], errors="coerce")
        n_bad = int(work[col].notna().sum() - converted.notna().sum())
        if n_bad > 0:
            non_numeric_notes.append(f"{col}:{n_bad}")
        work[col] = converted
    if non_numeric_notes:
        warnings.append(
            "Coerced non-numeric values to NaN in: " + ", ".join(non_numeric_notes)
        )

    strategy = (impute_strategy or "drop").lower()
    if strategy in {"mean", "median"}:
        for col in predictors:
            series = work[col]
            if not series.notna().any():
                raise ValueError(
                    f"Feature '{col}' is entirely NA after numeric coercion; "
                    "remove it from MethodBrief.features."
                )
            fill = float(series.mean() if strategy == "mean" else series.median())
            n_na = int(series.isna().sum())
            if n_na:
                work[col] = series.fillna(fill)
                warnings.append(
                    f"Imputed {n_na} NA in '{col}' with {strategy}={fill:.6g}."
                )
        before = len(work)
        work = work.dropna(subset=[target])
        dropped = before - len(work)
        if dropped:
            warnings.append(f"Dropped {dropped} rows with NA in target '{target}'.")
    else:
        before = len(work)
        work = work.dropna()
        dropped = before - len(work)
        if dropped:
            warnings.append(
                f"Dropped {dropped} rows with NA in analysis columns "
                f"(impute_strategy={strategy!r}, dropna={dropna})."
            )

    if len(work) < 20:
        raise ValueError(
            f"Insufficient rows after cleaning: n={len(work)}. "
            "Try impute_strategy='median' or reduce feature list."
        )

    x = work[predictors]
    y = work[target]
    if task_type == "classification":
        if y.dtype == object or str(y.dtype).startswith("string"):
            y = y.astype("category").cat.codes
            warnings.append("Encoded classification target labels to integer codes.")
        y = y.astype(int)
        if y.nunique() < 2:
            raise ValueError(
                "Classification target has fewer than 2 classes after cleaning."
            )
    return x, y, work, warnings


def format_tool_error(
    exc: BaseException,
    *,
    brief: MethodBrief | None = None,
    available_columns: list[str] | None = None,
    stage: str = "analysis",
) -> dict[str, Any]:
    """Empaqueta traceback + hints para auto-corrección del agente."""
    message = str(exc)
    suggestions: list[str] = []
    lower = message.lower()
    if "columns not found" in lower:
        suggestions.append(
            "Fix MethodBrief.features/target to match available_columns exactly."
        )
        suggestions.append(
            "Call Validate And Preprocess Dataset first to inspect columns."
        )
    if "insufficient rows" in lower:
        suggestions.append("Set impute_strategy to 'mean' or 'median'.")
        suggestions.append("Remove sparse features or reduce test_size.")
    if "not allowed for" in lower or "unsupported model" in lower:
        suggestions.append(
            "Use only whitelist models for the task_type "
            "(regression: ols/ridge/lasso/elasticnet/random_forest_reg; "
            "classification: logistic/random_forest_clf/gradient_boosting_clf)."
        )
    if "openpyxl" in lower:
        suggestions.append("Install openpyxl or export the dataset to CSV.")
    if not suggestions:
        suggestions.append(
            "Adjust MethodBrief parameters (features, impute_strategy, models, "
            "hyperparameter_grids) and retry Run Quantitative Analysis."
        )

    return {
        "status": "error",
        "stage": stage,
        "error_type": type(exc).__name__,
        "message": message,
        "traceback": traceback.format_exc(),
        "suggested_fixes": suggestions,
        "available_columns": available_columns or [],
        "brief_title": brief.title if brief else "",
        "dataset_path": brief.dataset_path if brief else "",
        "retry_hint": (
            "SELF-CORRECTION: update method_brief_json using suggested_fixes / "
            "available_columns, then call Run Quantitative Analysis again. "
            "Do NOT invent Python code; only edit MethodBrief fields."
        ),
        "model_results": [],
        "warnings": [f"{stage} error: {message}"],
    }


def _dataset_schema(df: pd.DataFrame, used: pd.DataFrame) -> DatasetSchema:
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    return DatasetSchema(
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns],
        dtypes=dtypes,
        missing_counts=missing,
        rows_after_dropna=int(len(used)),
    )


def _descriptive_table(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "variable": col,
                "n": int(series.notna().sum()),
                "mean": float(series.mean()) if series.notna().any() else None,
                "std": float(series.std()) if series.notna().any() else None,
                "min": float(series.min()) if series.notna().any() else None,
                "p50": float(series.median()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
            }
        )
    return rows


def _expand_grid(
    grid: dict[str, list[Any]],
    *,
    max_combinations: int,
) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [list(grid[k]) if grid[k] else [None] for k in keys]
    combos: list[dict[str, Any]] = []
    for product in itertools.product(*values):
        combos.append(dict(zip(keys, product, strict=True)))
        if len(combos) >= max_combinations:
            break
    return combos or [{}]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str,
) -> dict[str, float]:
    # binary average requires 2 classes; fallback to weighted if needed
    avg = average
    n_classes = len(np.unique(y_true))
    if avg == "binary" and n_classes != 2:
        avg = "weighted"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def _score_is_better(candidate: float, best: float | None, *, maximize: bool) -> bool:
    if best is None or (isinstance(best, float) and math.isnan(best)):
        return True
    return candidate > best if maximize else candidate < best


def _run_vif(x: pd.DataFrame) -> list[dict[str, Any]]:
    x_num = x.apply(pd.to_numeric, errors="coerce").dropna()
    if x_num.shape[1] < 2 or len(x_num) < x_num.shape[1] + 2:
        return []
    rows: list[dict[str, Any]] = []
    for i, col in enumerate(x_num.columns):
        try:
            vif = float(variance_inflation_factor(x_num.values, i))
        except Exception:  # noqa: BLE001
            vif = float("nan")
        rows.append({"variable": str(col), "vif": vif})
    return rows


def _residual_diagnostics(ols_model: Any) -> dict[str, Any]:
    resid = np.asarray(ols_model.resid)
    exog = ols_model.model.exog
    jb_stat, jb_p, _, _ = jarque_bera(resid)
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)
    except Exception as exc:  # noqa: BLE001
        return {
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_p),
            "breusch_pagan_error": str(exc),
        }
    return {
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_pvalue": float(bp_p),
    }


def run_quantitative_analysis(
    brief: MethodBrief,
    *,
    settings: Settings | None = None,
) -> AnalysisReport:
    """Ejecuta el motor determinista según el MethodBrief."""
    warnings: list[str] = []
    cfg = settings or get_settings()

    lit_snapshot, lit_warnings = load_literature_context(brief, settings=cfg)
    warnings.extend(lit_warnings)
    benchmarks = list(brief.literature_benchmarks)
    literature_benchmarking: list[dict[str, Any]] = []

    # Validar compatibilidad modelo ↔ task_type (pca se maneja aparte)
    supervised_models = [m for m in brief.models if m != "pca"]
    for model_id in supervised_models:
        if brief.task_type == "regression" and model_id not in ALLOWED_REGRESSION_MODELS:
            raise ValueError(
                f"Model {model_id!r} is not allowed for regression. "
                f"Allowed: {list(ALLOWED_REGRESSION_MODELS)}"
            )
        if (
            brief.task_type == "classification"
            and model_id not in ALLOWED_CLASSIFICATION_MODELS
        ):
            raise ValueError(
                f"Model {model_id!r} is not allowed for classification. "
                f"Allowed: {list(ALLOWED_CLASSIFICATION_MODELS)}"
            )
    if not supervised_models:
        raise ValueError(
            "MethodBrief.models must include at least one supervised model "
            "(ols/ridge/...); 'pca' alone is not sufficient."
        )

    df_raw, resolved = load_local_dataset(
        brief.dataset_path, sheet_name=brief.sheet_name, settings=cfg
    )
    predictors = brief.predictor_columns()
    x, y, work, prep_warnings = preprocess_analysis_frame(
        df_raw,
        target=brief.target,
        predictors=predictors,
        impute_strategy=brief.impute_strategy,
        dropna=brief.dropna,
        task_type=brief.task_type,
    )
    warnings.extend(prep_warnings)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=brief.test_size,
        random_state=brief.random_state,
        stratify=y if brief.task_type == "classification" and y.nunique() > 1 else None,
    )

    pca_extras: dict[str, Any] = {}
    if brief.run_pca or "pca" in brief.models:
        pca_result = run_pca(
            x_train,
            x_test,
            n_components=brief.pca_n_components,
            random_state=brief.random_state,
        )
        if pca_result.status != "ok":
            warnings.append(
                f"PCA wrapper failed: {pca_result.error_type}: {pca_result.message}"
            )
        else:
            pca_extras = {
                "n_components": pca_result.extras.get("n_components"),
                "explained_variance_ratio": pca_result.extras.get(
                    "explained_variance_ratio"
                ),
                "cumulative_explained_variance": pca_result.extras.get(
                    "cumulative_explained_variance"
                ),
                "loadings": pca_result.extras.get("loadings"),
            }
            if brief.run_pca:
                x_train = pca_result.extras["x_train_pca"]
                x_test = pca_result.extras["x_test_pca"]
                warnings.append(
                    f"Applied PCA before supervised models "
                    f"(n_components={pca_extras.get('n_components')})."
                )

    maximize = brief.metric_direction() == "maximize"
    iteration_log: list[dict[str, Any]] = []
    model_results: list[ModelResult] = []
    global_best_model: str | None = None
    global_best_score: float | None = None
    global_best_payload: dict[str, Any] | None = None

    remaining_budget = brief.max_grid_combinations

    for model_id in supervised_models:
        grid = brief.hyperparameter_grids.get(model_id, {})
        combos = _expand_grid(grid, max_combinations=max(1, remaining_budget))
        remaining_budget = max(0, remaining_budget - len(combos))
        if remaining_budget == 0 and model_id != supervised_models[-1]:
            warnings.append(
                "max_grid_combinations reached; remaining models may use default params only."
            )

        best_for_model: float | None = None
        best_pack: dict[str, Any] | None = None
        n_iter = 0

        for params in combos:
            n_iter += 1
            try:
                wrapped = dispatch_supervised_wrapper(
                    model_id,
                    x_train,
                    y_train,
                    x_test,
                    params=params,
                    random_state=brief.random_state,
                )
                if wrapped.status != "ok":
                    warnings.append(
                        f"{model_id} params={params} failed: "
                        f"{wrapped.error_type}: {wrapped.message}"
                    )
                    iteration_log.append(
                        {
                            "model_id": model_id,
                            "params": params,
                            "status": "error",
                            "error_type": wrapped.error_type,
                            "message": wrapped.message,
                            "traceback": wrapped.traceback,
                        }
                    )
                    continue

                y_tr_hat = wrapped.y_train_pred
                y_te_hat = wrapped.y_test_pred
                assert y_tr_hat is not None and y_te_hat is not None
                if brief.task_type == "regression":
                    metrics_train = _regression_metrics(
                        np.asarray(y_train), np.asarray(y_tr_hat)
                    )
                    metrics_test = _regression_metrics(
                        np.asarray(y_test), np.asarray(y_te_hat)
                    )
                    if wrapped.extras.get("aic") is not None:
                        metrics_test["aic"] = float(wrapped.extras["aic"])
                    if wrapped.extras.get("bic") is not None:
                        metrics_test["bic"] = float(wrapped.extras["bic"])
                else:
                    metrics_train = _classification_metrics(
                        np.asarray(y_train),
                        np.asarray(y_tr_hat),
                        average=brief.average,
                    )
                    metrics_test = _classification_metrics(
                        np.asarray(y_test),
                        np.asarray(y_te_hat),
                        average=brief.average,
                    )

                coefs = wrapped.coefficients
                importances = wrapped.feature_importances
                estimator = wrapped.estimator

                # Optional CV on training fold only (wrappers again)
                cv_mean = None
                cv_std = None
                if brief.cv_folds and brief.cv_folds >= 2 and model_id != "ols":
                    kf = KFold(
                        n_splits=brief.cv_folds,
                        shuffle=True,
                        random_state=brief.random_state,
                    )
                    fold_scores: list[float] = []
                    for tr_idx, va_idx in kf.split(x_train):
                        fold = dispatch_supervised_wrapper(
                            model_id,
                            x_train.iloc[tr_idx],
                            y_train.iloc[tr_idx],
                            x_train.iloc[va_idx],
                            params=params,
                            random_state=brief.random_state,
                        )
                        if fold.status != "ok" or fold.y_test_pred is None:
                            continue
                        pred = fold.y_test_pred
                        if brief.task_type == "regression":
                            m = _regression_metrics(
                                np.asarray(y_train.iloc[va_idx]), np.asarray(pred)
                            )
                        else:
                            m = _classification_metrics(
                                np.asarray(y_train.iloc[va_idx]),
                                np.asarray(pred),
                                average=brief.average,
                            )
                        fold_scores.append(float(m[brief.primary_metric]))
                    if fold_scores:
                        cv_mean = float(np.mean(fold_scores))
                        cv_std = float(np.std(fold_scores))

                score = float(metrics_test[brief.primary_metric])
                bench_rows: list[dict[str, Any]] = []
                for bench in benchmarks:
                    m_name = bench.metric.strip().lower()
                    if m_name not in metrics_test:
                        continue
                    bench_rows.extend(
                        compare_metric_to_benchmarks(
                            metric=m_name,
                            value=float(metrics_test[m_name]),
                            benchmarks=[bench],
                            model_id=model_id,
                            params=params,
                        )
                    )
                literature_benchmarking.extend(bench_rows)
                iteration_log.append(
                    {
                        "model_id": model_id,
                        "params": params,
                        "status": "ok",
                        "primary_metric": brief.primary_metric,
                        "score_test": score,
                        "metrics_test": metrics_test,
                        "cv_mean": cv_mean,
                        "literature_comparison": bench_rows,
                        "wrapper": model_id,
                    }
                )

                pack = {
                    "params": params,
                    "metrics_train": metrics_train,
                    "metrics_test": metrics_test,
                    "coefficients": coefs,
                    "feature_importances": importances,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "estimator": estimator,
                    "y_test_pred": np.asarray(y_te_hat),
                }
                if _score_is_better(score, best_for_model, maximize=maximize):
                    best_for_model = score
                    best_pack = pack
                if _score_is_better(score, global_best_score, maximize=maximize):
                    global_best_score = score
                    global_best_model = model_id
                    global_best_payload = {"model_id": model_id, **pack}
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{model_id} params={params} failed: {exc}")
                logger.warning("Model iteration failed: %s", exc)
                iteration_log.append(
                    {
                        "model_id": model_id,
                        "params": params,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        if best_pack is None:
            warnings.append(f"No successful runs for model {model_id}")
            continue

        model_results.append(
            ModelResult(
                model_id=model_id,
                best_params=best_pack["params"],
                metrics_train=best_pack["metrics_train"],
                metrics_test=best_pack["metrics_test"],
                cv_mean=best_pack["cv_mean"],
                cv_std=best_pack["cv_std"],
                coefficients=best_pack["coefficients"],
                feature_importances=best_pack["feature_importances"],
                n_iterations=n_iter,
            )
        )

    # Robustness on global best
    robustness: dict[str, Any] = {}
    if pca_extras:
        robustness["pca"] = pca_extras
    if global_best_payload is not None:
        if brief.robustness.vif and not brief.run_pca:
            robustness["vif"] = _run_vif(x)
        if (
            brief.task_type == "regression"
            and brief.robustness.residual_diagnostics
            and global_best_model == "ols"
        ):
            robustness["residual_diagnostics"] = _residual_diagnostics(
                global_best_payload["estimator"]
            )
        if brief.task_type == "classification":
            y_pred = global_best_payload["y_test_pred"]
            if brief.robustness.confusion_matrix:
                labels = sorted(set(np.asarray(y_test).tolist()))
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                robustness["confusion_matrix"] = {
                    "labels": [int(x) for x in labels],
                    "matrix": cm.astype(int).tolist(),
                }
            if brief.robustness.classification_report:
                robustness["classification_metrics_test"] = _classification_metrics(
                    np.asarray(y_test),
                    np.asarray(y_pred),
                    average=brief.average,
                )

    metrics_table = [
        {
            "model_id": m.model_id,
            "best_params": json.dumps(m.best_params, ensure_ascii=False),
            **{f"test_{k}": v for k, v in m.metrics_test.items()},
            "cv_mean": m.cv_mean,
            "n_iterations": m.n_iterations,
        }
        for m in model_results
    ]
    coef_table: list[dict[str, Any]] = []
    for m in model_results:
        source = m.coefficients or m.feature_importances
        for name, value in source.items():
            coef_table.append(
                {
                    "model_id": m.model_id,
                    "term": name,
                    "value": value,
                    "kind": "coefficient" if m.coefficients else "importance",
                }
            )

    report_bits = {
        "best_model": global_best_model,
        "best_score": global_best_score,
    }
    discussion = _deterministic_discussion(
        brief, report_bits, lit_snapshot, literature_benchmarking
    )
    if brief.enrich_discussion_with_llm and (
        lit_snapshot.get("evidence") or brief.literature_benchmarks
    ):
        discussion, llm_warnings = _llm_enrich_discussion(
            discussion,
            brief=brief,
            lit=lit_snapshot,
            benchmarking=literature_benchmarking,
            report_bits=report_bits,
        )
        warnings.extend(llm_warnings)

    # Deduplicate benchmarking rows preferring best model summaries at end
    bench_table = literature_benchmarking

    report = AnalysisReport(
        brief_title=brief.title,
        task_type=brief.task_type,
        dataset_path=str(resolved),
        dataset_schema=_dataset_schema(df_raw, work),
        primary_metric=brief.primary_metric,
        best_model=global_best_model,
        best_score=global_best_score,
        model_results=model_results,
        descriptive_tables={
            "summary_statistics": _descriptive_table(work, list(work.columns)),
        },
        robustness_tests=robustness,
        iteration_log=iteration_log,
        tables={
            "model_comparison": metrics_table,
            "coefficients_or_importances": coef_table,
            "literature_benchmarking": bench_table,
        },
        literature_snapshot=lit_snapshot,
        literature_benchmarking=bench_table,
        analytical_discussion=discussion,
        warnings=warnings,
    )
    return report


def report_to_markdown(report: AnalysisReport) -> str:
    """Renderiza el AnalysisReport en Markdown para el Redactor."""
    lines = [
        f"# Quantitative Analysis: {report.brief_title}",
        "",
        f"- Task: `{report.task_type}`",
        f"- Dataset: `{report.dataset_path}`",
        f"- Primary metric: `{report.primary_metric}`",
        f"- Best model: `{report.best_model}` (score={report.best_score})",
        f"- N (raw/clean): {report.dataset_schema.n_rows} / "
        f"{report.dataset_schema.rows_after_dropna}",
        "",
        "## Model comparison",
        "",
    ]
    comparison = report.tables.get("model_comparison") or []
    if not comparison:
        lines.append("_No model results._")
    else:
        headers = list(comparison[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in comparison:
            lines.append(
                "| "
                + " | ".join(str(row.get(h, "")) for h in headers)
                + " |"
            )
    lines.extend(["", "## Coefficients / feature importances", ""])
    coefs = report.tables.get("coefficients_or_importances") or []
    if not coefs:
        lines.append("_None._")
    else:
        lines.append("| model_id | term | value | kind |")
        lines.append("| --- | --- | --- | --- |")
        for row in coefs[:80]:
            lines.append(
                f"| {row.get('model_id')} | {row.get('term')} | "
                f"{row.get('value')} | {row.get('kind')} |"
            )

    if report.robustness_tests:
        lines.extend(["", "## Robustness tests", "", "```json"])
        lines.append(json.dumps(report.robustness_tests, ensure_ascii=False, indent=2))
        lines.append("```")

    if report.literature_benchmarking:
        lines.extend(["", "## Literature benchmarking", ""])
        lines.append(
            "| model_id | metric | observed | literature | delta | verdict | source |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in report.literature_benchmarking[:40]:
            lines.append(
                f"| {row.get('model_id')} | {row.get('metric')} | "
                f"{row.get('observed')} | {row.get('literature_reference')} | "
                f"{row.get('delta_observed_minus_ref')} | {row.get('verdict')} | "
                f"{row.get('source')} |"
            )

    if report.analytical_discussion:
        lines.extend(["", report.analytical_discussion.strip(), ""])

    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        for w in report.warnings:
            lines.append(f"- {w}")

    lines.append("")
    return "\n".join(lines)


@tool("Execute Python Analysis")
def execute_python_analysis_tool(
    dataset_path: str,
    python_code: str,
    sheet_name: str = "",
    timeout_seconds: float = 90.0,
) -> str:
    """Execute dynamic Python analysis code on a local CSV/XLSX DataFrame.

    Scientific namespace already includes: df (loaded copy), pd, np, sklearn,
    scipy, statsmodels/sm, xgboost/xgb (if installed). Assign metrics/tables to
    a dict named ``results``. On failure returns full traceback for
    self-correction — fix the script and call this tool again.

    Security: no network/shell; filesystem limited to data/workspace/output.

    Args:
        dataset_path: Local CSV/XLSX under data/workspace/output.
        python_code: Python script tailored to the methodological brief.
        sheet_name: Optional Excel sheet name (empty = first sheet).
        timeout_seconds: Soft execution timeout (default 90).

    Returns:
        JSON with status ok|error, stdout, results, traceback (if error).
    """
    settings = get_settings()
    try:
        df, resolved = load_local_dataset(
            dataset_path,
            sheet_name=sheet_name or None,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "stdout": "",
                "results": {},
                "retry_hint": (
                    "SELF-CORRECTION: fix dataset_path / sheet_name so the file "
                    "loads from the local sandbox, then re-run."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    try:
        payload = execute_analysis_code(
            df=df,
            code=python_code,
            dataset_path=str(resolved),
            settings=settings,
            timeout_seconds=timeout_seconds,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.exception("dynamic python analysis failed")
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "stdout": "",
                "results": {},
                "columns": [str(c) for c in df.columns],
                "retry_hint": (
                    "SELF-CORRECTION: read traceback, revise python_code, retry "
                    "Execute Python Analysis."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )


@tool("Validate And Preprocess Dataset")
def validate_and_preprocess_dataset_tool(method_brief_json: str) -> str:
    """Validate local CSV/XLSX columns and preview safe preprocessing.

    Does not train models. Use this before or after an analysis error to inspect
    available columns, missingness, and imputation effects. Returns JSON with
    status ok|error, available_columns, missing_columns, preprocess_warnings,
    and row counts. On error includes traceback and suggested_fixes for
    self-correction.

    Args:
        method_brief_json: MethodBrief JSON (dataset_path, target, features, …).
    """
    brief: MethodBrief | None = None
    available: list[str] = []
    try:
        raw = json.loads(method_brief_json)
        brief = MethodBrief.model_validate(raw)
        df, resolved = load_local_dataset(
            brief.dataset_path, sheet_name=brief.sheet_name
        )
        available = [str(c) for c in df.columns]
        predictors = brief.predictor_columns()
        missing = validate_columns(df, target=brief.target, features=predictors)
        if missing:
            raise ValueError(
                f"Columns not found in dataset: {missing}. "
                f"Available columns: {available}"
            )
        x, y, work, prep_warnings = preprocess_analysis_frame(
            df,
            target=brief.target,
            predictors=predictors,
            impute_strategy=brief.impute_strategy,
            dropna=brief.dropna,
            task_type=brief.task_type,
        )
        return json.dumps(
            {
                "status": "ok",
                "dataset_path": str(resolved),
                "available_columns": available,
                "missing_columns": [],
                "n_rows_raw": int(len(df)),
                "n_rows_clean": int(len(work)),
                "n_features": int(x.shape[1]),
                "target": brief.target,
                "predictors": predictors,
                "impute_strategy": brief.impute_strategy,
                "preprocess_warnings": prep_warnings,
                "retry_hint": (
                    "Preprocess OK. Call Run Quantitative Analysis with the same "
                    "or adjusted MethodBrief."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("validate/preprocess failed")
        payload = format_tool_error(
            exc, brief=brief, available_columns=available, stage="preprocess"
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)


@tool("Run Quantitative Analysis")
def run_quantitative_analysis_tool(method_brief_json: str) -> str:
    """Run deterministic quantitative analysis via pre-validated model wrappers.

    Does NOT let the LLM write Python. Parses MethodBrief and calls typed
    wrappers (OLS, Ridge, Lasso, ElasticNet, RandomForest, Logistic, GBM, PCA).
    On failure returns status=error with traceback and suggested_fixes so the
    agent can self-correct MethodBrief fields and retry.

    Args:
        method_brief_json: MethodBrief as JSON string.

    Returns:
        AnalysisReport JSON (status implicit ok) or structured error JSON.
    """
    brief: MethodBrief | None = None
    available: list[str] = []
    try:
        raw = json.loads(method_brief_json)
        brief = MethodBrief.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            format_tool_error(exc, brief=None, available_columns=[], stage="parse"),
            ensure_ascii=False,
            indent=2,
        )

    try:
        # Best-effort column listing for error payloads.
        try:
            df_preview, _ = load_local_dataset(
                brief.dataset_path, sheet_name=brief.sheet_name
            )
            available = [str(c) for c in df_preview.columns]
        except Exception:  # noqa: BLE001
            available = []

        report = run_quantitative_analysis(brief)
        payload = report.to_dict()
        payload["status"] = "ok"
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.exception("quantitative analysis failed")
        return json.dumps(
            format_tool_error(
                exc, brief=brief, available_columns=available, stage="analysis"
            ),
            ensure_ascii=False,
            indent=2,
        )
