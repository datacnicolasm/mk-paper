"""Entrypoint principal del sistema multi-agente."""

import logging

from mk_paper.config.settings import get_settings
from mk_paper.runtime import ensure_directories, setup_logging


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
    logger.info("Modelo rápido: %s", settings.litellm_fast_model)
    logger.info("Groq configurado: %s", bool(settings.groq_api_key))
    logger.info("Workspace: %s", settings.workspace_dir)
    logger.info("Output: %s", settings.output_dir)
    logger.info("Listo para orquestar el crew (implementación pendiente)")
    logger.info(
        "Prueba embudo sistemático: "
        "python -m mk_paper.cli literature --brief /app/data/briefs/example_volatility.json"
    )


if __name__ == "__main__":
    main()
