"""Utilidades de runtime compartidas (logging y directorios)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


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
