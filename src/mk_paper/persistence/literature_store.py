"""Persistencia local de resultados de búsqueda bibliográfica."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mk_paper.models.research_brief import LiteratureReviewOutput
from mk_paper.tools.systematic_review import review_to_markdown

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class LiteratureArtifacts:
    """Rutas de los artefactos persistidos de una búsqueda cruda."""

    run_dir: Path
    json_path: Path
    csv_path: Path
    latest_json: Path
    latest_csv: Path


@dataclass(frozen=True)
class ReviewArtifacts:
    """Rutas de los artefactos del embudo sistemático clasificado."""

    run_dir: Path
    json_path: Path
    md_path: Path
    latest_json: Path
    latest_md: Path


def _slugify(text: str, max_len: int = 48) -> str:
    """Convierte una query en slug seguro para nombres de archivo."""
    slug = _SLUG_RE.sub("-", text.lower().strip()).strip("-")
    return (slug or "query")[:max_len]


def _literature_root(output_dir: str | Path) -> Path:
    root = Path(output_dir) / "literature"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_literature_results(
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
    query: str,
) -> LiteratureArtifacts:
    """Persiste el JSON de búsqueda cruda y un CSV tabular."""
    root = _literature_root(output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{_slugify(query)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "results.json"
    csv_path = run_dir / "results.csv"
    latest_json = root / "latest.json"
    latest_csv = root / "latest.csv"

    enriched = {
        **payload,
        "persisted_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
    }
    json_text = json.dumps(enriched, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")

    papers = payload.get("papers") or []
    if papers:
        rows = []
        for paper in papers:
            rows.append(
                {
                    "doi": paper.get("doi"),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "citation_count": paper.get("citation_count"),
                    "is_oa": paper.get("is_oa"),
                    "oa_status": paper.get("oa_status"),
                    "pdf_url": paper.get("pdf_url"),
                    "landing_url": paper.get("landing_url"),
                    "venue": paper.get("venue"),
                    "authors": "; ".join(paper.get("authors") or []),
                    "sources": "; ".join(paper.get("sources") or []),
                }
            )
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=[
                "doi",
                "title",
                "year",
                "citation_count",
                "is_oa",
                "oa_status",
                "pdf_url",
                "landing_url",
                "venue",
                "authors",
                "sources",
            ]
        )

    df.to_csv(csv_path, index=False)
    df.to_csv(latest_csv, index=False)

    logger.info("Resultados crudos guardados en %s", run_dir)
    return LiteratureArtifacts(
        run_dir=run_dir,
        json_path=json_path,
        csv_path=csv_path,
        latest_json=latest_json,
        latest_csv=latest_csv,
    )


def save_literature_review(
    output: LiteratureReviewOutput | dict[str, Any],
    *,
    output_dir: str | Path,
) -> ReviewArtifacts:
    """Persiste review.json + review.md clasificado (Core vs Conceptual)."""
    import shutil

    if isinstance(output, dict):
        review = LiteratureReviewOutput.model_validate(output)
    else:
        review = output

    root = _literature_root(output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{_slugify(review.brief_title)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = run_dir / "pdfs"
    text_dir = run_dir / "fulltext"
    pdf_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)

    # Copiar PDFs locales y guardar excerpts de texto de papers core.
    for paper in [
        *review.core_findings,
        *review.conceptual_references,
        *review.seminal_literature,
    ]:
        slug = _slugify(paper.doi or paper.title or "paper")
        if paper.pdf_local_path:
            src = Path(paper.pdf_local_path)
            if src.exists():
                dest = pdf_dir / f"{slug}.pdf"
                try:
                    shutil.copy2(src, dest)
                    paper.pdf_local_path = str(dest)
                except OSError as exc:
                    logger.warning("Could not copy PDF %s: %s", src, exc)
        if paper.full_text_excerpt:
            (text_dir / f"{slug}.txt").write_text(
                paper.full_text_excerpt, encoding="utf-8"
            )

    payload = review.to_dict()
    payload["persisted_at"] = datetime.now(timezone.utc).isoformat()
    payload["run_id"] = run_dir.name

    json_path = run_dir / "review.json"
    md_path = run_dir / "review.md"
    latest_json = root / "latest_review.json"
    latest_md = root / "latest_review.md"

    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    md_text = review_to_markdown(review)

    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    logger.info("Review sistemático guardado en %s", run_dir)
    return ReviewArtifacts(
        run_dir=run_dir,
        json_path=json_path,
        md_path=md_path,
        latest_json=latest_json,
        latest_md=latest_md,
    )
