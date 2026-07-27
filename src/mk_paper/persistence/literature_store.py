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

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class LiteratureArtifacts:
    """Rutas de los artefactos persistidos de una búsqueda."""

    run_dir: Path
    json_path: Path
    csv_path: Path
    latest_json: Path
    latest_csv: Path


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
    """Persiste el JSON de búsqueda y un CSV tabular para inspección rápida.

    Estructura::

        output/literature/
          YYYYMMDDTHHMMSSZ_slug/
            results.json
            results.csv
          latest.json   # copia del último JSON
          latest.csv    # copia del último CSV

    Args:
        payload: Diccionario con query, count, papers, warnings.
        output_dir: Directorio base de salida de la app.
        query: Términos de búsqueda (para el nombre del run).

    Returns:
        Rutas de los artefactos generados.
    """
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

    logger.info("Resultados guardados en %s", run_dir)
    return LiteratureArtifacts(
        run_dir=run_dir,
        json_path=json_path,
        csv_path=csv_path,
        latest_json=latest_json,
        latest_csv=latest_csv,
    )
