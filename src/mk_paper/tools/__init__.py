"""Tools del sistema multi-agente."""

from mk_paper.tools.analysis_tools import (
    execute_python_analysis_tool,
    run_quantitative_analysis_tool,
    validate_and_preprocess_dataset_tool,
)
from mk_paper.tools.auditor_tools import (
    evaluate_manuscript_quality_tool,
    polish_final_manuscript_tool,
    run_audit_with_feedback_loop_tool,
    run_structural_quality_checks_tool,
)
from mk_paper.tools.literature_tools import search_scientific_literature
from mk_paper.tools.systematic_review import run_systematic_literature_review
from mk_paper.tools.writer_tools import (
    build_citation_catalog_tool,
    draft_imrad_paper_tool,
    export_paper_formats_tool,
    load_writing_inputs_tool,
    validate_apa_citations_tool,
)

__all__ = [
    "search_scientific_literature",
    "run_systematic_literature_review",
    "execute_python_analysis_tool",
    "run_quantitative_analysis_tool",
    "validate_and_preprocess_dataset_tool",
    "load_writing_inputs_tool",
    "build_citation_catalog_tool",
    "draft_imrad_paper_tool",
    "validate_apa_citations_tool",
    "export_paper_formats_tool",
    "run_structural_quality_checks_tool",
    "evaluate_manuscript_quality_tool",
    "run_audit_with_feedback_loop_tool",
    "polish_final_manuscript_tool",
]
