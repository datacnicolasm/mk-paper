"""Subcomando CLI para probar la tool de búsqueda bibliográfica."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from mk_paper.config.settings import get_settings
from mk_paper.persistence.literature_store import save_literature_results
from mk_paper.runtime import ensure_directories, setup_logging
from mk_paper.tools.literature_tools import _run_literature_search

logger = logging.getLogger(__name__)


def add_literature_parser(parser: argparse.ArgumentParser) -> None:
    """Registra argumentos del subcomando ``literature``."""
    parser.add_argument(
        "query",
        type=str,
        help="Términos de búsqueda académica (ej. 'volatility forecasting LSTM').",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=5,
        help="Máximo de papers a retornar (1-50, default: 5).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Solo imprime en consola; no persiste JSON/CSV.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Imprime el JSON completo en consola (default: activo).",
    )
    parser.add_argument(
        "--quiet-json",
        action="store_true",
        help="No imprime el JSON completo; solo el resumen tabular.",
    )


def _print_summary(payload: dict[str, Any]) -> None:
    """Imprime un resumen legible de los papers encontrados."""
    papers = payload.get("papers") or []
    warnings = payload.get("warnings") or []

    print()
    print("=" * 72)
    print(f"Query : {payload.get('query')}")
    print(f"Count : {payload.get('count', 0)}")
    print(f"Source: {payload.get('primary_source') or 'n/a'}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print("=" * 72)

    if not papers:
        print("No se encontraron papers.")
        return

    for idx, paper in enumerate(papers, start=1):
        title = paper.get("title") or "(sin título)"
        year = paper.get("year") or "?"
        doi = paper.get("doi") or "N/A"
        cites = paper.get("citation_count") or 0
        oa = "OA" if paper.get("is_oa") else "closed"
        pdf = paper.get("pdf_url") or "—"
        print(f"\n[{idx}] {title}")
        print(f"    year={year}  cites={cites}  access={oa}")
        print(f"    doi={doi}")
        print(f"    pdf={pdf}")


def run_literature_search_cli(args: argparse.Namespace) -> int:
    """Ejecuta la búsqueda, muestra resultado y persiste artefactos."""
    settings = get_settings()
    setup_logging(settings.log_level)
    ensure_directories(
        settings.data_dir,
        settings.workspace_dir,
        settings.output_dir,
    )

    limit = max(1, min(int(args.limit), 50))
    logger.info("Iniciando búsqueda aislada: query=%r limit=%s", args.query, limit)

    raw = asyncio.run(_run_literature_search(args.query, limit))
    payload: dict[str, Any] = json.loads(raw)

    _print_summary(payload)

    if not args.quiet_json:
        print("\n--- JSON ---")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if not args.no_save:
        artifacts = save_literature_results(
            payload,
            output_dir=settings.output_dir,
            query=args.query,
        )
        print("\n--- Persistencia ---")
        print(f"run_dir     : {artifacts.run_dir}")
        print(f"results.json: {artifacts.json_path}")
        print(f"results.csv : {artifacts.csv_path}")
        print(f"latest.json : {artifacts.latest_json}")
        print(f"latest.csv  : {artifacts.latest_csv}")
    else:
        logger.info("Persistencia omitida (--no-save).")

    return 0 if (payload.get("count") or 0) > 0 or not payload.get("warnings") else 0
