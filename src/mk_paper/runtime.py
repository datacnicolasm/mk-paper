"""Utilidades de runtime compartidas (logging, directorios y sandbox de paths)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mk_paper.config.settings import Settings, get_settings


def setup_logging(level: str) -> None:
    """Configura el logging de la aplicación."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def ensure_directories(*paths: str) -> None:
    """Crea directorios de trabajo si no existen."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_sandbox_path(
    raw_path: str,
    *,
    settings: Settings | None = None,
    must_exist: bool = True,
) -> Path:
    """Resuelve un path bajo data/workspace/output (sandbox de lectura)."""
    cfg = settings or get_settings()
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        for base in (
            Path(cfg.data_dir),
            Path(cfg.workspace_dir),
            Path(cfg.output_dir),
            Path.cwd(),
        ):
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
