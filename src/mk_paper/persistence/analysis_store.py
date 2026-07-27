"""Persistencia local de reportes de análisis cuantitativo."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mk_paper.models.method_brief import AnalysisReport
from mk_paper.tools.analysis_tools import report_to_markdown

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Rutas de los artefactos persistidos del análisis."""

    run_dir: Path
    json_path: Path
    md_path: Path
    metrics_csv: Path
    coef_csv: Path
    latest_json: Path
    latest_md: Path


def _slugify(text: str, max_len: int = 48) -> str:
    slug = _SLUG_RE.sub("-", text.lower().strip()).strip("-")
    return (slug or "analysis")[:max_len]


def _analysis_root(output_dir: str | Path) -> Path:
    root = Path(output_dir) / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_analysis_report(
    output: AnalysisReport | dict[str, Any],
    *,
    output_dir: str | Path,
) -> AnalysisArtifacts:
    """Persiste report.json, report.md y CSV de tablas."""
    if isinstance(output, dict):
        report = AnalysisReport.model_validate(output)
    else:
        report = output

    root = _analysis_root(output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{_slugify(report.brief_title)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "report.json"
    md_path = run_dir / "report.md"
    metrics_csv = run_dir / "model_comparison.csv"
    coef_csv = run_dir / "coefficients.csv"
    latest_json = root / "latest_report.json"
    latest_md = root / "latest_report.md"

    payload = report.to_dict()
    payload["persisted_at"] = datetime.now(timezone.utc).isoformat()
    payload["run_id"] = run_dir.name

    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    md_text = report_to_markdown(report)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    comparison = report.tables.get("model_comparison") or []
    coefs = report.tables.get("coefficients_or_importances") or []
    pd.DataFrame(comparison).to_csv(metrics_csv, index=False)
    pd.DataFrame(coefs).to_csv(coef_csv, index=False)

    logger.info("Analysis report saved to %s", run_dir)
    return AnalysisArtifacts(
        run_dir=run_dir,
        json_path=json_path,
        md_path=md_path,
        metrics_csv=metrics_csv,
        coef_csv=coef_csv,
        latest_json=latest_json,
        latest_md=latest_md,
    )
