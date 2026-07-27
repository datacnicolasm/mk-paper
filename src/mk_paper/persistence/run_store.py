"""Persistencia de ejecuciones end-to-end bajo ``output/runs/``."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class PipelineRunContext:
    """Carpeta única de una corrida del pipeline."""

    run_id: str
    run_dir: Path
    literature_dir: Path
    analysis_dir: Path
    paper_dir: Path
    audit_dir: Path
    briefs_dir: Path
    final_dir: Path
    logs_dir: Path
    manifest_path: Path
    log_path: Path
    _log_lines: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{stamp}] {message}"
        self._log_lines.append(line)
        logger.info("%s", message)
        self.log_path.write_text("\n".join(self._log_lines) + "\n", encoding="utf-8")

    def write_json(self, relative: str, payload: dict[str, Any] | list[Any]) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def write_text(self, relative: str, text: str) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "", encoding="utf-8")
        return path

    def copy_file(self, src: Path, relative_dest: str) -> Path:
        dest = self.run_dir / relative_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest


def _slugify(text: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("-", (text or "").lower().strip()).strip("-")
    return (slug or "paper_run")[:max_len]


def create_pipeline_run(
    *,
    output_dir: str | Path,
    title: str = "paper_run",
) -> PipelineRunContext:
    """Crea ``output/runs/{timestamp}_{slug}/`` con subcarpetas estándar."""
    root = Path(output_dir) / "runs"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{_slugify(title)}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    literature_dir = run_dir / "literature"
    analysis_dir = run_dir / "analysis"
    paper_dir = run_dir / "paper"
    audit_dir = run_dir / "audit"
    briefs_dir = run_dir / "briefs"
    final_dir = run_dir / "final"
    logs_dir = run_dir / "logs"
    for d in (
        literature_dir,
        analysis_dir,
        paper_dir,
        audit_dir,
        briefs_dir,
        final_dir,
        logs_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    ctx = PipelineRunContext(
        run_id=run_id,
        run_dir=run_dir,
        literature_dir=literature_dir,
        analysis_dir=analysis_dir,
        paper_dir=paper_dir,
        audit_dir=audit_dir,
        briefs_dir=briefs_dir,
        final_dir=final_dir,
        logs_dir=logs_dir,
        manifest_path=run_dir / "manifest.json",
        log_path=logs_dir / "pipeline.log",
    )
    ctx.log(f"Pipeline run created: {run_dir}")
    return ctx


def write_manifest(ctx: PipelineRunContext, payload: dict[str, Any]) -> Path:
    """Escribe/actualiza el manifiesto de la corrida."""
    enriched = {
        **payload,
        "run_id": ctx.run_id,
        "run_dir": str(ctx.run_dir),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    ctx.manifest_path.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ctx.manifest_path
