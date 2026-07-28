"""Persistencia local de dictámenes del Quality Auditor."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mk_paper.models.audit_brief import AuditVerdict

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class AuditArtifacts:
    """Rutas de artefactos del audit."""

    run_dir: Path
    verdict_json: Path
    polished_md: Path
    polished_tex: Path | None
    latest_verdict: Path
    latest_md: Path
    latest_tex: Path | None


def _slugify(text: str, max_len: int = 48) -> str:
    slug = _SLUG_RE.sub("-", text.lower().strip()).strip("-")
    return (slug or "audit")[:max_len]


def _audit_root(output_dir: str | Path) -> Path:
    root = Path(output_dir) / "audit"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_audit_verdict(
    verdict: AuditVerdict | dict[str, Any],
    *,
    output_dir: str | Path,
) -> AuditArtifacts:
    """Persiste review_verdict.json + markdown/LaTeX pulidos."""
    if isinstance(verdict, dict):
        v = AuditVerdict.model_validate(verdict)
    else:
        v = verdict

    root = _audit_root(output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{_slugify(v.title)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    verdict_json = run_dir / "review_verdict.json"
    polished_md = run_dir / "manuscript_final.md"
    latest_verdict = root / "review_verdict.json"
    latest_md = root / "manuscript_final.md"

    polished = v.polished_markdown or ""
    payload = v.to_dict()
    payload["persisted_at"] = datetime.now(timezone.utc).isoformat()
    payload["run_id"] = run_dir.name

    verdict_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    latest_verdict.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    polished_md.write_text(polished, encoding="utf-8")
    latest_md.write_text(polished, encoding="utf-8")

    polished_tex: Path | None = None
    latest_tex: Path | None = None
    if v.latex:
        polished_tex = run_dir / "manuscript_final.tex"
        latest_tex = root / "manuscript_final.tex"
        polished_tex.write_text(v.latex, encoding="utf-8")
        latest_tex.write_text(v.latex, encoding="utf-8")

    logger.info("Audit verdict saved under %s", run_dir)
    return AuditArtifacts(
        run_dir=run_dir,
        verdict_json=verdict_json,
        polished_md=polished_md,
        polished_tex=polished_tex,
        latest_verdict=latest_verdict,
        latest_md=latest_md,
        latest_tex=latest_tex,
    )
