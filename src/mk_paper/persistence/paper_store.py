"""Persistencia local de borradores IMRaD del Scientific Writer."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mk_paper.models.writing_brief import CitationCatalog, PaperDraft

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class PaperArtifacts:
    """Rutas de los artefactos persistidos del paper."""

    run_dir: Path
    draft_md: Path
    draft_tex: Path | None
    draft_json: Path
    catalog_json: Path
    latest_md: Path
    latest_json: Path
    latest_tex: Path | None


def _slugify(text: str, max_len: int = 48) -> str:
    slug = _SLUG_RE.sub("-", text.lower().strip()).strip("-")
    return (slug or "paper")[:max_len]


def _paper_root(output_dir: str | Path) -> Path:
    root = Path(output_dir) / "paper"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_paper_draft(
    draft: PaperDraft | dict[str, Any],
    catalog: CitationCatalog | dict[str, Any] | None,
    *,
    output_dir: str | Path,
) -> PaperArtifacts:
    """Persiste draft_imrad.md/.tex, paper_draft.json y citation_catalog.json."""
    if isinstance(draft, dict):
        paper = PaperDraft.model_validate(draft)
    else:
        paper = draft

    if isinstance(catalog, dict):
        cat = CitationCatalog.model_validate(catalog)
    elif catalog is None:
        cat = CitationCatalog()
    else:
        cat = catalog

    root = _paper_root(output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{_slugify(paper.title)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    draft_md = run_dir / "draft_imrad.md"
    draft_json = run_dir / "paper_draft.json"
    catalog_json = run_dir / "citation_catalog.json"
    latest_md = root / "draft_imrad.md"
    latest_json = root / "latest_paper.json"

    payload = paper.to_dict()
    payload["persisted_at"] = datetime.now(timezone.utc).isoformat()
    payload["run_id"] = run_dir.name

    draft_md.write_text(paper.markdown or "", encoding="utf-8")
    draft_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    catalog_json.write_text(
        json.dumps(cat.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latest_md.write_text(paper.markdown or "", encoding="utf-8")
    latest_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    draft_tex: Path | None = None
    latest_tex: Path | None = None
    if paper.latex:
        draft_tex = run_dir / "draft_imrad.tex"
        latest_tex = root / "draft_imrad.tex"
        draft_tex.write_text(paper.latex, encoding="utf-8")
        latest_tex.write_text(paper.latex, encoding="utf-8")

    logger.info("Paper draft saved under %s", run_dir)
    return PaperArtifacts(
        run_dir=run_dir,
        draft_md=draft_md,
        draft_tex=draft_tex,
        draft_json=draft_json,
        catalog_json=catalog_json,
        latest_md=latest_md,
        latest_json=latest_json,
        latest_tex=latest_tex,
    )
