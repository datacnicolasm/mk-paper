"""Entrypoint principal del sistema multi-agente."""

import logging

from mk_paper.config.settings import get_settings
from mk_paper.runtime import ensure_directories, setup_logging


def main() -> None:
    """Punto de entrada del paquete (mensaje + tip de pipeline)."""
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
    logger.info(
        "Pipeline E2E: python -m mk_paper run-pipeline "
        "--research-brief data/briefs/example_research.json "
        "--method-brief data/briefs/example_method.json "
        "--dataset data/workspace/pymes_colombia.csv"
    )
    logger.info(
        "Componentes: python -m mk_paper.cli {literature|analysis|paper|audit}"
    )


if __name__ == "__main__":
    main()
