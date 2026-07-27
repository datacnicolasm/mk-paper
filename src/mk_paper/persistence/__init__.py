"""Capa de persistencia local de artefactos del proyecto."""

from mk_paper.persistence.literature_store import (
    LiteratureArtifacts,
    save_literature_results,
)

__all__ = ["LiteratureArtifacts", "save_literature_results"]
