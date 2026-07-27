"""Entrypoint principal del sistema multi-agente."""

import logging
import sys
from pathlib import Path

from mk_paper.config.settings import get_settings


def setup_logging(level: str) -> None:
    """Configura el logging de la aplicación."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def ensure_directories(*paths: str) -> None:
    """Crea directorios de trabajo si no existen."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Punto de entrada del crew."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    ensure_directories(
        settings.data_dir,
        settings.workspace_dir,
        settings.output_dir,
    )

    logger.info("mk-paper iniciado")
    logger.info("Modelo principal: %s", settings.litellm_model)
    logger.info("Workspace: %s", settings.workspace_dir)
    logger.info("Output: %s", settings.output_dir)
    logger.info("Listo para orquestar el crew (implementación pendiente)")


if __name__ == "__main__":
    main()
