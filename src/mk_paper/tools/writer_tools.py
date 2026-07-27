"""Tools del Scientific Writer: APA 7, IMRaD, anti-alucinación de citas."""

from __future__ import annotations

import json
import logging
import re
import traceback
import unicodedata
from typing import Any

from crewai.tools import tool

from mk_paper.config.llm import get_llm
from mk_paper.config.settings import Settings, get_settings
from mk_paper.models.method_brief import AnalysisReport
from mk_paper.models.research_brief import ClassifiedPaper, LiteratureReviewOutput
from mk_paper.models.writing_brief import (
    CitationCatalog,
    CitationEntry,
    CitationValidation,
    PaperDraft,
    WritingBrief,
)
from mk_paper.tools.analysis_tools import resolve_sandbox_path

logger = logging.getLogger(__name__)

# Pandoc-style citations only: [@key] or [@key1; @key2]
_CITE_KEY_RE = re.compile(r"\[@([\w-]+)\]")
_CITE_CLUSTER_RE = re.compile(r"\[((?:@[\w-]+)(?:\s*;\s*@[\w-]+)*)\]")
_CITE_KEY_TOKEN_RE = re.compile(r"@([\w-]+)")
_REF_SECTION_RE = re.compile(
    r"(?im)^#{1,3}\s*Referencias\s*$|^#{1,3}\s*References\s*$"
)

TABLE_MODEL_COMPARISON = "{{TABLE_MODEL_COMPARISON}}"
TABLE_COEFFICIENTS = "{{TABLE_COEFFICIENTS}}"
TABLE_ROBUSTNESS = "{{TABLE_ROBUSTNESS}}"
TABLE_BENCHMARKING = "{{TABLE_BENCHMARKING}}"


# ---------------------------------------------------------------------------
# APA helpers
# ---------------------------------------------------------------------------


def _surname_initials(full_name: str) -> tuple[str, str]:
    """Separa apellido e iniciales estilo APA a partir de 'First M. Last'."""
    parts = [p for p in re.split(r"\s+", (full_name or "").strip()) if p]
    if not parts:
        return "Unknown", ""
    if len(parts) == 1:
        return parts[0], ""
    surname = parts[-1].rstrip(",")
    given = parts[:-1]
    initials = " ".join(
        (g[0].upper() + ".") for g in given if g and g[0].isalpha()
    )
    return surname, initials


def _apa_author_list(authors: list[str]) -> str:
    """Formatea lista de autores para referencia APA 7."""
    formatted: list[str] = []
    for name in authors:
        surname, initials = _surname_initials(name)
        if initials:
            formatted.append(f"{surname}, {initials}")
        else:
            formatted.append(surname)
    n = len(formatted)
    if n == 0:
        return ""
    if n == 1:
        return formatted[0]
    if n == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def _intext_author_phrase(authors: list[str], *, narrative: bool) -> str:
    """Fragmento de autores para cita parentética o narrativa."""
    surnames = [_surname_initials(a)[0] for a in authors if a.strip()]
    if not surnames:
        return "Unknown"
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        joiner = " and " if narrative else " & "
        return f"{surnames[0]}{joiner}{surnames[1]}"
    return f"{surnames[0]} et al."


def format_apa_reference(
    *,
    authors: list[str],
    year: int | None,
    title: str | None,
    venue: str | None,
    doi: str | None,
) -> str:
    """Construye una referencia APA 7 (journal-like) a partir de metadatos."""
    author_part = _apa_author_list(authors) or "Anonymous"
    year_part = str(year) if year is not None else "n.d."
    title_part = (title or "Untitled").rstrip(".")
    pieces = [f"{author_part} ({year_part}). {title_part}."]
    if venue:
        pieces.append(f" *{venue.strip()}*.")
    if doi:
        doi_clean = doi.replace("https://doi.org/", "").strip()
        pieces.append(f" https://doi.org/{doi_clean}")
    return "".join(pieces)


def format_apa_parenthetical(authors: list[str], year: int | None) -> str:
    y = str(year) if year is not None else "n.d."
    return f"({_intext_author_phrase(authors, narrative=False)}, {y})"


def format_apa_narrative(authors: list[str], year: int | None) -> str:
    y = str(year) if year is not None else "n.d."
    return f"{_intext_author_phrase(authors, narrative=True)} ({y})"


def _ascii_slug(text: str) -> str:
    """Normaliza acentos para cite_keys estables (Gómez → gomez)."""
    normalized = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", stripped.lower())


def _make_cite_key(
    authors: list[str],
    year: int | None,
    *,
    used_keys: set[str],
) -> str:
    """Clave estilo Pandoc estable: surname+year (sufijo solo si hay colisión)."""
    surname = _surname_initials(authors[0])[0] if authors else "anon"
    slug = _ascii_slug(surname) or "anon"
    y = str(year) if year is not None else "nd"
    base = f"{slug}{y}"
    if base not in used_keys:
        return base
    n = 2
    while f"{base}_{n}" in used_keys:
        n += 1
    return f"{base}_{n}"


def _paper_dedupe_key(paper: ClassifiedPaper | dict[str, Any]) -> str:
    if isinstance(paper, ClassifiedPaper):
        doi = (paper.doi or "").strip().lower()
        title = (paper.title or "").strip().lower()
        year = paper.year
    else:
        doi = str(paper.get("doi") or "").strip().lower()
        title = str(paper.get("title") or "").strip().lower()
        year = paper.get("year")
    if doi:
        return f"doi:{doi}"
    return f"title:{title}|year:{year}"


def build_citation_catalog(
    review: LiteratureReviewOutput | dict[str, Any],
) -> CitationCatalog:
    """Une seminal/core/conceptual y genera APA 7 (sin inventar metadatos)."""
    if isinstance(review, dict):
        lit = LiteratureReviewOutput.model_validate(review)
    else:
        lit = review

    buckets: list[tuple[str, ClassifiedPaper]] = []
    for p in lit.seminal_literature:
        buckets.append(("seminal", p))
    for p in lit.core_findings:
        buckets.append(("core", p))
    for p in lit.conceptual_references:
        buckets.append(("conceptual", p))

    seen: set[str] = set()
    entries: list[CitationEntry] = []
    warnings: list[str] = []
    used_keys: set[str] = set()

    for level, paper in buckets:
        dkey = _paper_dedupe_key(paper)
        if dkey in seen:
            continue
        seen.add(dkey)

        authors = [str(a).strip() for a in (paper.authors or []) if str(a).strip()]
        if not authors or paper.year is None:
            warnings.append(
                "Skipped citation (missing authors or year): "
                f"{paper.title or paper.doi or 'unknown'}"
            )
            continue

        cite_key = _make_cite_key(authors, paper.year, used_keys=used_keys)
        used_keys.add(cite_key)

        resolved_level = level  # type: ignore[assignment]
        if paper.level in ("core", "conceptual", "seminal"):
            resolved_level = paper.level

        entry = CitationEntry(
            cite_key=cite_key,
            authors=authors,
            year=paper.year,
            title=paper.title,
            venue=paper.venue,
            doi=paper.doi,
            level=resolved_level,  # type: ignore[arg-type]
            key_findings=list(paper.key_findings or []),
            citation_context=paper.citation_context or "",
            suggested_section=paper.suggested_section or "",
            apa_reference=format_apa_reference(
                authors=authors,
                year=paper.year,
                title=paper.title,
                venue=paper.venue,
                doi=paper.doi,
            ),
            apa_parenthetical=format_apa_parenthetical(authors, paper.year),
            apa_narrative=format_apa_narrative(authors, paper.year),
        )
        entries.append(entry)

    by_key = {e.cite_key: e for e in entries}
    return CitationCatalog(entries=entries, by_key=by_key, warnings=warnings)


def extract_pandoc_cite_keys(markdown: str) -> list[str]:
    """Extrae cite_keys Pandoc (`[@key]` / `[@a; @b]`) en orden de aparición."""
    body = _strip_references_section(markdown)
    found: list[str] = []
    seen: set[str] = set()
    for cluster in _CITE_CLUSTER_RE.finditer(body):
        for token in _CITE_KEY_TOKEN_RE.finditer(cluster.group(1)):
            key = token.group(1)
            if key not in seen:
                seen.add(key)
                found.append(key)
    for m in _CITE_KEY_RE.finditer(body):
        key = m.group(1)
        if key not in seen:
            seen.add(key)
            found.append(key)
    return found


def expand_pandoc_citations(
    markdown: str, catalog: CitationCatalog
) -> tuple[str, list[str]]:
    """Reemplaza clusters Pandoc por APA parentética determinista.

    - ``[@beneish1999]`` → ``(Beneish, 1999)``
    - ``[@beneish1999; @piotroski2000]`` → ``(Beneish, 1999; Piotroski, 2000)``
    """
    used: list[str] = []
    by_key = catalog.by_key or {e.cite_key: e for e in catalog.entries}

    def _replace_cluster(match: re.Match[str]) -> str:
        inner = match.group(1)
        keys = _CITE_KEY_TOKEN_RE.findall(inner)
        apa_parts: list[str] = []
        for key in keys:
            used.append(key)
            entry = by_key.get(key)
            if entry is None:
                apa_parts.append(f"@{key}")
                continue
            phrase = _intext_author_phrase(entry.authors, narrative=False)
            year = str(entry.year) if entry.year is not None else "n.d."
            apa_parts.append(f"{phrase}, {year}")
        if not apa_parts:
            return match.group(0)
        if any(p.startswith("@") for p in apa_parts):
            return match.group(0)
        return "(" + "; ".join(apa_parts) + ")"

    expanded = _CITE_CLUSTER_RE.sub(_replace_cluster, markdown or "")
    ordered: list[str] = []
    seen: set[str] = set()
    for k in used:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return expanded, ordered


def _bibliography_markdown(
    catalog: CitationCatalog,
    used_keys: list[str] | None,
    *,
    cite_all: bool,
    heading: str,
) -> str:
    if cite_all or not used_keys:
        selected = list(catalog.entries)
    else:
        key_set = set(used_keys)
        selected = [e for e in catalog.entries if e.cite_key in key_set]

    selected = sorted(
        selected,
        key=lambda e: (
            _surname_initials(e.authors[0])[0].lower() if e.authors else e.cite_key
        ),
    )
    lines = [f"## {heading}", ""]
    for entry in selected:
        lines.append(entry.apa_reference)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_apa_references(
    markdown: str,
    catalog: CitationCatalog,
    *,
    cite_all: bool = False,
    heading: str = "Referencias",
    used_keys: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Expande ``[@cite_key]`` a APA y añade la sección Referencias (determinista).

    Returns:
        Tuple (markdown_final, cite_keys_usados).
    """
    body = _strip_references_section(markdown)
    expanded, inferred = expand_pandoc_citations(body, catalog)
    keys = list(used_keys) if used_keys else inferred
    if cite_all:
        keys = [e.cite_key for e in catalog.entries]
    merged: list[str] = []
    seen: set[str] = set()
    for k in [*keys, *inferred]:
        if k not in seen:
            seen.add(k)
            merged.append(k)
    refs = _bibliography_markdown(
        catalog, merged, cite_all=cite_all, heading=heading
    )
    return expanded.rstrip() + "\n\n" + refs, merged


# ---------------------------------------------------------------------------
# Load inputs + factual skeleton
# ---------------------------------------------------------------------------


def load_writing_inputs(
    brief: WritingBrief,
    *,
    settings: Settings | None = None,
) -> tuple[LiteratureReviewOutput, AnalysisReport, list[str]]:
    """Carga review.json + AnalysisReport desde sandbox local."""
    cfg = settings or get_settings()
    warnings: list[str] = []

    lit_path = resolve_sandbox_path(
        brief.literature_review_path, settings=cfg, must_exist=True
    )
    ana_path = resolve_sandbox_path(
        brief.analysis_report_path, settings=cfg, must_exist=True
    )

    lit_raw = json.loads(lit_path.read_text(encoding="utf-8"))
    # Strip persistence extras that may break validation
    for drop in ("persisted_at", "run_id"):
        lit_raw.pop(drop, None)
    review = LiteratureReviewOutput.model_validate(lit_raw)

    ana_raw = json.loads(ana_path.read_text(encoding="utf-8"))
    for drop in ("persisted_at", "run_id"):
        ana_raw.pop(drop, None)
    report = AnalysisReport.model_validate(ana_raw)

    if not review.core_findings and not review.seminal_literature and not review.conceptual_references:
        warnings.append("Literature review has no citable papers.")
    if not report.model_results:
        warnings.append("Analysis report has no model_results.")

    return review, report, warnings


def _metrics_table_md(report: AnalysisReport) -> str:
    rows = []
    headers = [
        "model_id",
        "best_params",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_accuracy",
        "test_f1",
        "cv_mean",
    ]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for m in report.model_results:
        mt = m.metrics_test or {}
        rows.append(
            "| "
            + " | ".join(
                [
                    m.model_id,
                    json.dumps(m.best_params, ensure_ascii=False),
                    str(mt.get("rmse", "")),
                    str(mt.get("mae", "")),
                    str(mt.get("r2", "")),
                    str(mt.get("accuracy", "")),
                    str(mt.get("f1", mt.get("f1_weighted", ""))),
                    str(m.cv_mean if m.cv_mean is not None else ""),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _coef_table_md(report: AnalysisReport) -> str:
    lines = ["| model_id | term | value | kind |", "| --- | --- | --- | --- |"]
    for m in report.model_results:
        for term, val in (m.coefficients or {}).items():
            lines.append(f"| {m.model_id} | {term} | {val} | coefficient |")
        for term, val in (m.feature_importances or {}).items():
            lines.append(f"| {m.model_id} | {term} | {val} | importance |")
    if len(lines) == 2:
        return "_No coefficients or feature importances reported._"
    return "\n".join(lines)


def build_factual_skeleton(
    report: AnalysisReport,
    brief: WritingBrief,
    catalog: CitationCatalog,
) -> dict[str, Any]:
    """Bloques factuales (números solo del AnalysisReport)."""
    schema = report.dataset_schema
    cite_cards = []
    for e in catalog.entries:
        cite_cards.append(
            {
                "cite_key": e.cite_key,
                "pandoc": f"[@{e.cite_key}]",
                "level": e.level,
                "apa_parenthetical": e.apa_parenthetical,
                "apa_narrative": e.apa_narrative,
                "apa_reference": e.apa_reference,
                "key_findings": e.key_findings,
                "citation_context": e.citation_context,
                "suggested_section": e.suggested_section,
                "doi": e.doi,
            }
        )
    robustness_md = ""
    if report.robustness_tests:
        robustness_md = (
            "```json\n"
            + json.dumps(report.robustness_tests, ensure_ascii=False, indent=2)
            + "\n```"
        )
    benchmarking_md = ""
    if report.literature_benchmarking:
        benchmarking_md = (
            "```json\n"
            + json.dumps(
                report.literature_benchmarking[:20], ensure_ascii=False, indent=2
            )
            + "\n```"
        )
    return {
        "manuscript_title": brief.title,
        "manuscript_authors": brief.authors,
        "language": brief.language,
        "analysis_title": report.brief_title,
        "task_type": report.task_type,
        "dataset_path": report.dataset_path,
        "n_rows": schema.n_rows,
        "n_rows_clean": schema.rows_after_dropna,
        "columns": schema.columns,
        "primary_metric": report.primary_metric,
        "best_model": report.best_model,
        "best_score": report.best_score,
        "model_comparison_markdown": _metrics_table_md(report),
        "coefficients_markdown": _coef_table_md(report),
        "robustness_markdown": robustness_md,
        "benchmarking_markdown": benchmarking_md,
        "robustness_tests": report.robustness_tests,
        "literature_benchmarking": report.literature_benchmarking,
        "analytical_discussion": report.analytical_discussion,
        "analysis_warnings": report.warnings,
        "citation_cards": cite_cards,
        "table_placeholders": {
            "model_comparison": TABLE_MODEL_COMPARISON,
            "coefficients": TABLE_COEFFICIENTS,
            "robustness": TABLE_ROBUSTNESS,
            "benchmarking": TABLE_BENCHMARKING,
        },
    }


# ---------------------------------------------------------------------------
# Citation validation (Pandoc keys only — no APA regex)
# ---------------------------------------------------------------------------


def _strip_references_section(markdown: str) -> str:
    match = _REF_SECTION_RE.search(markdown or "")
    if not match:
        return markdown or ""
    return (markdown or "")[: match.start()].rstrip()


def validate_draft_citations(
    markdown: str,
    catalog: CitationCatalog,
) -> CitationValidation:
    """Valida únicamente cite_keys Pandoc ``[@key]`` / ``[@a; @b]`` vs catálogo."""
    body = _strip_references_section(markdown)
    allowed = set(catalog.allowed_keys())
    found = extract_pandoc_cite_keys(body)
    ok = [k for k in found if k in allowed]
    unknown = [k for k in found if k not in allowed]
    # También marcar tokens sueltos [@...] capturados por el patrón simple
    for m in _CITE_KEY_RE.finditer(body):
        key = m.group(1)
        if key not in found:
            found.append(key)
            if key in allowed:
                ok.append(key)
            else:
                unknown.append(key)
    # Deduplicate
    def _uniq(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    found_u, ok_u, unk_u = _uniq(found), _uniq(ok), _uniq(unknown)
    return CitationValidation(
        status="ok" if not unk_u else "error",
        citations_found=found_u,
        citations_ok=ok_u,
        citations_unknown=unk_u,
        message=(
            "All Pandoc cite_keys match the literature catalog."
            if not unk_u
            else f"Unknown cite_keys (not in catalog): {unk_u}"
        ),
    )


def inject_literal_tables(body: str, skeleton: dict[str, Any]) -> str:
    """Inyecta tablas Markdown exactas del skeleton (el LLM no las reescribe)."""
    text = body or ""
    replacements = {
        TABLE_MODEL_COMPARISON: str(skeleton.get("model_comparison_markdown") or ""),
        TABLE_COEFFICIENTS: str(skeleton.get("coefficients_markdown") or ""),
        TABLE_ROBUSTNESS: str(skeleton.get("robustness_markdown") or ""),
        TABLE_BENCHMARKING: str(skeleton.get("benchmarking_markdown") or ""),
    }
    for placeholder, value in replacements.items():
        if placeholder in text:
            text = text.replace(placeholder, value)
    return text


# ---------------------------------------------------------------------------
# Markdown → LaTeX (basic academic export)
# ---------------------------------------------------------------------------


def _latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    return "".join(out)


def markdown_to_latex(markdown: str, *, title: str = "") -> str:
    """Export tipográfico básico IMRaD (secciones + tablas GFM simples)."""
    lines = (markdown or "").splitlines()
    body: list[str] = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{hyperref}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{setspace}",
        r"\onehalfspacing",
        rf"\title{{{_latex_escape(title or 'Manuscript')}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        "",
    ]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s\-:|]+\|$", lines[i + 1]):
            # GFM table
            header = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            ncols = len(header)
            colspec = "l" * ncols
            body.append(r"\begin{tabular}{" + colspec + "}")
            body.append(r"\toprule")
            body.append(" & ".join(_latex_escape(h) for h in header) + r" \\")
            body.append(r"\midrule")
            for row in rows:
                padded = (row + [""] * ncols)[:ncols]
                body.append(" & ".join(_latex_escape(c) for c in padded) + r" \\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append("")
            continue
        if line.startswith("# "):
            body.append(r"\section*{" + _latex_escape(line[2:].strip()) + "}")
        elif line.startswith("## "):
            body.append(r"\section{" + _latex_escape(line[3:].strip()) + "}")
        elif line.startswith("### "):
            body.append(r"\subsection{" + _latex_escape(line[4:].strip()) + "}")
        elif line.strip() == "":
            body.append("")
        else:
            body.append(_latex_escape(line))
        i += 1
    body.append(r"\end{document}")
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# Draft orchestration
# ---------------------------------------------------------------------------


def _infer_used_keys(markdown: str, catalog: CitationCatalog) -> list[str]:
    validation = validate_draft_citations(markdown, catalog)
    return list(validation.citations_ok)


def _pandoc_cite(entries: list[CitationEntry], n: int = 2) -> str:
    """Emite cluster Pandoc ``[@a]`` o ``[@a; @b]``."""
    keys = [e.cite_key for e in entries[:n] if e.cite_key]
    if not keys:
        return ""
    if len(keys) == 1:
        return f"[@{keys[0]}]"
    inner = "; ".join(f"@{k}" for k in keys)
    return f"[{inner}]"


def _deterministic_imrad_body(
    skeleton: dict[str, Any],
    catalog: CitationCatalog,
    *,
    language: str,
) -> tuple[str, list[str]]:
    """Borrador factual sin LLM: cite_keys Pandoc + placeholders de tablas."""
    seminal = [e for e in catalog.entries if e.level == "seminal"]
    core = [e for e in catalog.entries if e.level == "core"]
    conceptual = [e for e in catalog.entries if e.level == "conceptual"]
    intro_pool = [*seminal, *conceptual]

    title = skeleton["manuscript_title"]
    cite_intro = _pandoc_cite(intro_pool, 3)
    cite_core = _pandoc_cite(core, 2)

    if language == "en":
        body = (
            f"# {title}\n\n"
            "## Introduction\n\n"
            "This study addresses longitudinal financial forensic audit signals for "
            "insolvency risk and accounting manipulation. The theoretical frame draws "
            f"on seminal and conceptual sources {cite_intro}. "
            "Objectives follow the local analysis brief without extrapolating beyond "
            "the curated sample.\n\n"
            "### Research gap and objectives\n\n"
            "Despite established forensic scores and machine-learning classifiers, "
            "transparent local baselines remain useful for methodological calibration. "
            f"We estimate the declared supervised models and compare test "
            f"`{skeleton['primary_metric']}` on the local dataset.\n\n"
            "## Method\n\n"
            f"Dataset: `{skeleton['dataset_path']}` "
            f"(N={skeleton['n_rows']}; N_clean={skeleton['n_rows_clean']}). "
            f"Task: `{skeleton['task_type']}`. Columns: "
            f"{', '.join(skeleton.get('columns') or [])}. "
            f"Primary metric: `{skeleton['primary_metric']}`. "
            "Estimators and hyperparameters are those recorded by the Quantitative "
            "Analyst (e.g., Random Forest and forensic score features when present).\n\n"
            "## Results\n\n"
            f"Best model by primary metric: `{skeleton['best_model']}` "
            f"(score={skeleton['best_score']}). Interpretation paragraphs below do not "
            "alter numeric tables; tables are injected verbatim from the analysis "
            "report.\n\n"
            "### Model comparison\n\n"
            f"{TABLE_MODEL_COMPARISON}\n\n"
            "The ranking should be read strictly from the injected table; no values "
            "are restated with alternative rounding.\n\n"
            "### Coefficients and feature importances\n\n"
            f"{TABLE_COEFFICIENTS}\n\n"
            "### Robustness checks\n\n"
            f"{TABLE_ROBUSTNESS}\n\n"
            "## Discussion\n\n"
            "Empirical benchmarking against core literature is limited to catalogued "
            f"Nivel-1 sources {cite_core}. "
            "Numeric claims remain confined to the local sample and model set.\n\n"
            "### Literature benchmarks\n\n"
            f"{TABLE_BENCHMARKING}\n\n"
        )
        if skeleton.get("analytical_discussion"):
            body += (
                "### Analyst notes\n\n"
                + str(skeleton["analytical_discussion"])[:4000]
                + "\n\n"
            )
        body += (
            "Limitations include sample coverage, feature scope, and regimes absent "
            "from the local file. No causal claims are advanced.\n"
        )
    else:
        body = (
            f"# {title}\n\n"
            "## Introducción\n\n"
            "Este estudio examina señales de auditoría forense financiera en un "
            "panel longitudinal de insolvencia y manipulación contable. El marco "
            "teórico se construye casi exclusivamente con literatura seminal y "
            f"conceptual {cite_intro}. "
            "Los objetivos siguen el brief analítico local sin extrapolar fuera de "
            "la muestra curada.\n\n"
            "### Vacío de conocimiento y objetivos\n\n"
            "Aunque existen scores forenses clásicos y clasificadores de "
            "aprendizaje automático, siguen siendo útiles baselines locales "
            "replicables. Se estiman los modelos declarados y se compara "
            f"`{skeleton['primary_metric']}` de prueba en el dataset local.\n\n"
            "## Metodología\n\n"
            f"Dataset: `{skeleton['dataset_path']}` "
            f"(N={skeleton['n_rows']}; N_limpio={skeleton['n_rows_clean']}). "
            f"Tarea: `{skeleton['task_type']}`. Columnas: "
            f"{', '.join(skeleton.get('columns') or [])}. "
            f"Métrica primaria: `{skeleton['primary_metric']}`. "
            "Los estimadores e hiperparámetros son los registrados por el Analista "
            "Cuantitativo (p. ej., Random Forest y features de scores forenses cuando "
            "apliquen).\n\n"
            "## Resultados\n\n"
            f"Mejor modelo según la métrica primaria: `{skeleton['best_model']}` "
            f"(score={skeleton['best_score']}). Los párrafos de interpretación no "
            "alteran las tablas numéricas; estas se inyectan literalmente desde el "
            "reporte de análisis.\n\n"
            "### Comparación de modelos\n\n"
            f"{TABLE_MODEL_COMPARISON}\n\n"
            "El ordenamiento debe leerse estrictamente desde la tabla inyectada; no "
            "se reescriben magnitudes con redondeo alternativo.\n\n"
            "### Coeficientes e importancias\n\n"
            f"{TABLE_COEFFICIENTS}\n\n"
            "### Pruebas de robustez\n\n"
            f"{TABLE_ROBUSTNESS}\n\n"
            "## Discusión\n\n"
            "El contraste empírico con la literatura se limita a fuentes *core* del "
            f"catálogo {cite_core}. "
            "Las afirmaciones numéricas permanecen acotadas a la muestra y al "
            "conjunto de modelos locales.\n\n"
            "### Benchmarks de literatura\n\n"
            f"{TABLE_BENCHMARKING}\n\n"
        )
        if skeleton.get("analytical_discussion"):
            body += (
                "### Notas del analista\n\n"
                + str(skeleton["analytical_discussion"])[:4000]
                + "\n\n"
            )
        body += (
            "Las limitaciones incluyen cobertura muestral, alcance de "
            "características y regímenes ausentes del archivo local. No se "
            "formulan afirmaciones causales.\n"
        )

    used = extract_pandoc_cite_keys(body)
    return body, used


def _llm_draft_imrad(
    skeleton: dict[str, Any],
    catalog: CitationCatalog,
    *,
    language: str,
    correction_feedback: str = "",
) -> tuple[str, list[str]]:
    """Solicita al LLM el cuerpo IMRaD con cite_keys Pandoc (sin Referencias)."""
    cite_block = json.dumps(
        skeleton.get("citation_cards") or [], ensure_ascii=False, indent=2
    )
    lang_name = "Spanish" if language == "es" else "English"
    section_names = (
        "Introducción, Metodología, Resultados, Discusión"
        if language == "es"
        else "Introduction, Method, Results, Discussion"
    )
    feedback = ""
    if correction_feedback:
        feedback = f"\nCORRECTION FEEDBACK FROM VALIDATOR:\n{correction_feedback}\n"

    seminal_keys = [e.cite_key for e in catalog.entries if e.level == "seminal"]
    conceptual_keys = [e.cite_key for e in catalog.entries if e.level == "conceptual"]
    core_keys = [e.cite_key for e in catalog.entries if e.level == "core"]

    # Skeleton without large table strings (placeholders only for the LLM)
    skeleton_for_llm = {
        k: v
        for k, v in skeleton.items()
        if k
        not in {
            "model_comparison_markdown",
            "coefficients_markdown",
            "robustness_markdown",
            "benchmarking_markdown",
            "citation_cards",
        }
    }

    prompt = f"""You are a Q1–Q2 academic scientific writer.

Write a rigorous IMRaD manuscript body in {lang_name}.

CITATION PROTOCOL (MANDATORY — Pandoc keys only):
- You MUST cite using ONLY pandoc cite_keys like [@beneish1999] or clusters [@beneish1999; @piotroski2000].
- NEVER write APA author-year prose citations yourself (no "Beneish (1999)", no "(Smith, 2020)").
- Software will deterministically expand [@cite_key] to APA and build Referencias.
- Allowed keys ONLY from CITATION CARDS below.

LITERATURE LEVEL MAPPING (HARD RULES):
- Introducción / marco teórico: cite almost exclusively level=seminal and level=conceptual
  (keys: {seminal_keys + conceptual_keys}).
- Discusión: use level=core for empirical benchmarking (keys: {core_keys}).
- Do NOT use core papers as the backbone of the Introduction.

RESULTS TABLES (HARD RULE):
- You are STRICTLY FORBIDDEN from typing, rounding, or reformatting numeric tables.
- In Resultados, insert these placeholders EXACTLY (verbatim tokens):
  {TABLE_MODEL_COMPARISON}
  {TABLE_COEFFICIENTS}
  {TABLE_ROBUSTNESS}
  {TABLE_BENCHMARKING}  (only if discussion needs benchmark block; else still allowed)
- Your prose may interpret statistical meaning, but must not restate altered figures.

OTHER RULES:
1. Scientific, objective tone. No marketing/hype.
2. Never invent papers, DOIs, authors, years, or metrics.
3. Markdown ## headings for: {section_names}
4. Do NOT write a Referencias/References section.
5. Methodology must describe the local dataset/brief/models from FACTUAL SKELETON.

{feedback}

MANUSCRIPT TITLE: {skeleton.get('manuscript_title')}
FACTUAL SKELETON (JSON; tables omitted — use placeholders):
{json.dumps(skeleton_for_llm, ensure_ascii=False, indent=2)[:12000]}

CITATION CARDS (ONLY allowed sources; use field "pandoc"):
{cite_block[:12000]}

Return ONLY the Markdown body (no JSON fences).
"""
    llm = get_llm()
    text = str(llm.call(prompt)).strip()
    text = _strip_references_section(text)
    used = extract_pandoc_cite_keys(text)
    return text, used


def assemble_paper_markdown(
    body: str,
    catalog: CitationCatalog,
    skeleton: dict[str, Any],
    used_keys: list[str],
    *,
    cite_all: bool,
    language: str,
) -> tuple[str, list[str]]:
    """Inyecta tablas literales, expande Pandoc→APA y añade Referencias."""
    heading = "Referencias" if language == "es" else "References"
    with_tables = inject_literal_tables(_strip_references_section(body), skeleton)
    # Ensure placeholders were replaced; if LLM omitted them, append tables under Results
    if TABLE_MODEL_COMPARISON in with_tables or TABLE_COEFFICIENTS in with_tables:
        with_tables = inject_literal_tables(with_tables, skeleton)
    if TABLE_MODEL_COMPARISON in with_tables:
        with_tables = with_tables.replace(
            TABLE_MODEL_COMPARISON,
            str(skeleton.get("model_comparison_markdown") or ""),
        )
    if TABLE_COEFFICIENTS in with_tables:
        with_tables = with_tables.replace(
            TABLE_COEFFICIENTS,
            str(skeleton.get("coefficients_markdown") or ""),
        )
    if TABLE_ROBUSTNESS in with_tables:
        with_tables = with_tables.replace(
            TABLE_ROBUSTNESS,
            str(skeleton.get("robustness_markdown") or ""),
        )
    if TABLE_BENCHMARKING in with_tables:
        with_tables = with_tables.replace(
            TABLE_BENCHMARKING,
            str(skeleton.get("benchmarking_markdown") or ""),
        )

    final_md, merged_keys = render_apa_references(
        with_tables,
        catalog,
        cite_all=cite_all,
        heading=heading,
        used_keys=used_keys,
    )
    return final_md, merged_keys


def draft_imrad_paper(
    brief: WritingBrief,
    *,
    settings: Settings | None = None,
    use_llm: bool = True,
) -> tuple[PaperDraft, CitationCatalog]:
    """Orquesta carga → catálogo → draft IMRaD → validación → LaTeX opcional."""
    cfg = settings or get_settings()
    warnings: list[str] = []

    review, report, load_warnings = load_writing_inputs(brief, settings=cfg)
    warnings.extend(load_warnings)

    catalog = build_citation_catalog(review)
    warnings.extend(catalog.warnings)

    skeleton = build_factual_skeleton(report, brief, catalog)
    lit_path = str(
        resolve_sandbox_path(brief.literature_review_path, settings=cfg)
    )
    ana_path = str(
        resolve_sandbox_path(brief.analysis_report_path, settings=cfg)
    )

    body = ""
    used_keys: list[str] = []
    llm_used = False

    if use_llm:
        try:
            body, used_keys = _llm_draft_imrad(
                skeleton, catalog, language=brief.language
            )
            llm_used = True
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"LLM draft skipped: {exc}")
            body, used_keys = _deterministic_imrad_body(
                skeleton, catalog, language=brief.language
            )
    else:
        body, used_keys = _deterministic_imrad_body(
            skeleton, catalog, language=brief.language
        )

    # Validar cite_keys Pandoc ANTES de expandir a APA
    validation = validate_draft_citations(body, catalog)

    # One self-correction pass if LLM invented cite_keys
    if (
        llm_used
        and validation.status == "error"
        and validation.citations_unknown
    ):
        try:
            feedback = (
                "Remove or replace unknown Pandoc cite_keys. Use ONLY:\n"
                + json.dumps(
                    [
                        {
                            "cite_key": e.cite_key,
                            "pandoc": f"[@{e.cite_key}]",
                            "level": e.level,
                        }
                        for e in catalog.entries
                    ],
                    ensure_ascii=False,
                    indent=2,
                )[:6000]
                + f"\nUnknown found: {validation.citations_unknown}"
            )
            body, used_keys = _llm_draft_imrad(
                skeleton,
                catalog,
                language=brief.language,
                correction_feedback=feedback,
            )
            validation = validate_draft_citations(body, catalog)
            if validation.status == "error":
                warnings.append(
                    "Citation validation still failing after one correction pass: "
                    + validation.message
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Citation self-correction skipped: {exc}")

    markdown, used_keys = assemble_paper_markdown(
        body,
        catalog,
        skeleton,
        used_keys or validation.citations_ok,
        cite_all=brief.cite_all_catalog,
        language=brief.language,
    )

    latex = None
    if brief.include_latex:
        latex = markdown_to_latex(markdown, title=brief.title)

    status: str = "ok"
    if validation.status == "error":
        status = "partial"
    if not catalog.entries:
        status = "error"
        warnings.append("Empty citation catalog; cannot produce APA-backed manuscript.")

    draft = PaperDraft(
        title=brief.title,
        markdown=markdown,
        latex=latex,
        citations_used=used_keys,
        warnings=warnings,
        validation=validation,
        literature_path=lit_path,
        analysis_path=ana_path,
        language=brief.language,
        status=status,  # type: ignore[arg-type]
    )
    return draft, catalog


def revise_imrad_with_feedback(
    brief: WritingBrief,
    feedback: dict[str, Any] | Any,
    previous_markdown: str,
    *,
    settings: Settings | None = None,
    use_llm: bool = True,
) -> tuple[PaperDraft, CitationCatalog]:
    """Reescribe el borrador aplicando AuditFeedback estructurado (Writer loop)."""
    from mk_paper.models.audit_brief import AuditFeedback as _AuditFeedback

    cfg = settings or get_settings()
    if isinstance(feedback, dict):
        fb = _AuditFeedback.model_validate(feedback)
    else:
        fb = feedback

    review, report, warnings = load_writing_inputs(brief, settings=cfg)
    catalog = build_citation_catalog(review)
    warnings.extend(catalog.warnings)
    skeleton = build_factual_skeleton(report, brief, catalog)

    cite_block = json.dumps(
        [
            {
                "cite_key": e.cite_key,
                "pandoc": f"[@{e.cite_key}]",
                "level": e.level,
            }
            for e in catalog.entries
        ],
        ensure_ascii=False,
        indent=2,
    )
    findings_block = json.dumps(
        [f.model_dump(mode="json") for f in fb.findings],
        ensure_ascii=False,
        indent=2,
    )[:8000]

    body = ""
    used_keys: list[str] = []
    if use_llm:
        try:
            prompt = f"""You are a Q1–Q2 Scientific Writer revising a manuscript after peer audit.

Apply ALL revision instructions. Return a full IMRaD Markdown body in {"Spanish" if brief.language == "es" else "English"}.

HARD RULES:
1. Continuous academic prose ONLY — NO bullet lists / numbered lists for objectives,
   research questions, justifications, or methods. Tables (GFM) are allowed.
2. Cite ONLY with Pandoc keys [@cite_key] or [@a; @b]. Never write Author (Year) yourself.
3. Introducción/marco teórico: seminal + conceptual keys. Discusión: core keys.
4. In Resultados insert these placeholders EXACTLY (do not retype numeric tables):
   {TABLE_MODEL_COMPARISON}
   {TABLE_COEFFICIENTS}
   {TABLE_ROBUSTNESS}
   {TABLE_BENCHMARKING}
5. Do NOT write Referencias (software appends APA).
6. Analytical, restrained tone — no marketing language.
7. Align problem statement, quantitative model, and conclusions with the analysis facts.

AUDIT SCORE: {fb.overall_score} / 10 (threshold {fb.threshold})
DECISION: {fb.decision}
SUMMARY: {fb.summary}
REVISION INSTRUCTIONS:
{fb.revision_instructions}

FINDINGS JSON:
{findings_block}

ALLOWED CITE KEYS:
{cite_block}

FACTUAL SKELETON (no tables — use placeholders):
{json.dumps({k: v for k, v in skeleton.items() if k not in {"model_comparison_markdown", "coefficients_markdown", "robustness_markdown", "benchmarking_markdown", "citation_cards"}}, ensure_ascii=False, indent=2)[:10000]}

PREVIOUS MARKDOWN (for context; rewrite fully):
{_strip_references_section(previous_markdown)[:12000]}

Return ONLY the revised Markdown body.
"""
            llm = get_llm()
            body = _strip_references_section(str(llm.call(prompt)).strip())
            used_keys = extract_pandoc_cite_keys(body)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"LLM revision skipped: {exc}")
            body, used_keys = _deterministic_imrad_body(
                skeleton, catalog, language=brief.language
            )
            # Soft deterministic fixes: strip bullet lines in prose sections
            body = _strip_structural_bullets(body)
    else:
        body, used_keys = _deterministic_imrad_body(
            skeleton, catalog, language=brief.language
        )
        body = _strip_structural_bullets(body)
        # Prepend a short revision note in prose if feedback present
        if fb.revision_instructions and brief.language == "es":
            note = (
                "Esta versión incorpora correcciones de auditoría de calidad "
                "orientadas a prosa continua, mesura analítica y alineación "
                "método–resultados–conclusiones.\n\n"
            )
            # Insert after title line
            parts = body.split("\n", 2)
            if len(parts) >= 2:
                body = parts[0] + "\n\n" + note + "\n".join(parts[1:])

    validation = validate_draft_citations(body, catalog)
    markdown, used_keys = assemble_paper_markdown(
        body,
        catalog,
        skeleton,
        used_keys or validation.citations_ok,
        cite_all=brief.cite_all_catalog,
        language=brief.language,
    )
    latex = (
        markdown_to_latex(markdown, title=brief.title) if brief.include_latex else None
    )
    draft = PaperDraft(
        title=brief.title,
        markdown=markdown,
        latex=latex,
        citations_used=used_keys,
        warnings=warnings,
        validation=validation,
        literature_path=str(
            resolve_sandbox_path(brief.literature_review_path, settings=cfg)
        ),
        analysis_path=str(
            resolve_sandbox_path(brief.analysis_report_path, settings=cfg)
        ),
        language=brief.language,
        status="ok" if validation.status == "ok" else "partial",
    )
    return draft, catalog


def _strip_structural_bullets(markdown: str) -> str:
    """Convierte viñetas estructurales a prosa mínima (fallback sin LLM)."""
    out: list[str] = []
    for line in (markdown or "").splitlines():
        if re.match(r"^\s*\|", line) or line.strip().startswith("```"):
            out.append(line)
            continue
        m = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if m:
            out.append(m.group(1).rstrip(".") + ".")
            continue
        m2 = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if m2:
            out.append(m2.group(1).rstrip(".") + ".")
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CrewAI tools
# ---------------------------------------------------------------------------


@tool("Load Writing Inputs")
def load_writing_inputs_tool(writing_brief_json: str) -> str:
    """Load local literature review.json and AnalysisReport for IMRaD writing.

    Args:
        writing_brief_json: WritingBrief JSON with literature_review_path and
            analysis_report_path under data/workspace/output.

    Returns:
        JSON with status, review summary, analysis summary, warnings.
    """
    try:
        brief = WritingBrief.model_validate(json.loads(writing_brief_json))
        review, report, warnings = load_writing_inputs(brief)
        return json.dumps(
            {
                "status": "ok",
                "title": brief.title,
                "literature": {
                    "brief_title": review.brief_title,
                    "n_seminal": len(review.seminal_literature),
                    "n_core": len(review.core_findings),
                    "n_conceptual": len(review.conceptual_references),
                },
                "analysis": {
                    "brief_title": report.brief_title,
                    "best_model": report.best_model,
                    "best_score": report.best_score,
                    "primary_metric": report.primary_metric,
                    "n_models": len(report.model_results),
                },
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )


@tool("Build Citation Catalog")
def build_citation_catalog_tool(writing_brief_json: str) -> str:
    """Build deterministic APA 7 citation catalog from the literature review only.

    Never invents authors/years/DOIs. Papers lacking authors or year are skipped
    with warnings.

    Args:
        writing_brief_json: WritingBrief JSON pointing to review.json.

    Returns:
        JSON CitationCatalog (entries with apa_reference / in-text forms).
    """
    try:
        brief = WritingBrief.model_validate(json.loads(writing_brief_json))
        review, _, warnings = load_writing_inputs(brief)
        catalog = build_citation_catalog(review)
        payload = catalog.model_dump(mode="json")
        # by_key duplicates entries; keep for agent lookup but trim size via keys list
        payload["allowed_keys"] = catalog.allowed_keys()
        payload["load_warnings"] = warnings
        payload["status"] = "ok"
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )


@tool("Draft IMRAD Paper")
def draft_imrad_paper_tool(writing_brief_json: str) -> str:
    """Draft an IMRaD scientific paper merging literature JSON + AnalysisReport.

    Uses APA 7 citations only from the literature catalog. Appends a deterministic
    Referencias section. On citation hallucinations, attempts one correction pass.

    Args:
        writing_brief_json: Full WritingBrief JSON.

    Returns:
        JSON PaperDraft (markdown, latex optional, validation, warnings).
    """
    try:
        brief = WritingBrief.model_validate(json.loads(writing_brief_json))
        draft, catalog = draft_imrad_paper(brief, use_llm=True)
        payload = draft.to_dict()
        payload["catalog_size"] = len(catalog.entries)
        payload["allowed_cite_keys"] = catalog.allowed_keys()
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.exception("draft_imrad_paper_tool failed")
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "markdown": "",
                "warnings": [],
            },
            ensure_ascii=False,
            indent=2,
        )


@tool("Validate APA Citations")
def validate_apa_citations_tool(
    markdown: str,
    writing_brief_json: str,
) -> str:
    """Validate Pandoc cite_keys ``[@key]`` / ``[@a; @b]`` against the catalog.

    Does NOT parse APA author-year strings. Unknown keys are returned for
    self-correction. APA expansion is performed later by render_apa_references.

    Args:
        markdown: Draft markdown still containing Pandoc cite_keys (body).
        writing_brief_json: WritingBrief to rebuild the allowed catalog.

    Returns:
        JSON CitationValidation with citations_unknown for self-correction.
    """
    try:
        brief = WritingBrief.model_validate(json.loads(writing_brief_json))
        review, _, _ = load_writing_inputs(brief)
        catalog = build_citation_catalog(review)
        # Validate body only for unknown DOI false positives from refs
        body = _strip_references_section(markdown)
        result = validate_draft_citations(body, catalog)
        return json.dumps(
            {"status": result.status, **result.model_dump(mode="json")},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )


@tool("Export Paper Formats")
def export_paper_formats_tool(
    markdown: str,
    title: str = "Manuscript",
    include_latex: str = "true",
) -> str:
    """Export manuscript Markdown and optional basic LaTeX.

    Args:
        markdown: Full IMRaD markdown including Referencias.
        title: Document title for LaTeX.
        include_latex: \"true\"/\"false\" string flag.

    Returns:
        JSON with markdown echo and latex string when requested.
    """
    try:
        flag = str(include_latex).strip().lower() in {"1", "true", "yes", "y"}
        latex = markdown_to_latex(markdown, title=title) if flag else None
        return json.dumps(
            {
                "status": "ok",
                "title": title,
                "markdown": markdown,
                "latex": latex,
                "include_latex": flag,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )
