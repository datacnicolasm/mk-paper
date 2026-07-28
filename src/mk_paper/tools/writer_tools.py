"""Tools del Expert Academic Writer: APA 7, Intro + Lit Review, anti-alucinación."""

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
from mk_paper.models.research_brief import ClassifiedPaper, LiteratureReviewOutput
from mk_paper.models.writing_brief import (
    CitationCatalog,
    CitationEntry,
    CitationValidation,
    PaperDraft,
    WritingBrief,
)
from mk_paper.runtime import resolve_sandbox_path

logger = logging.getLogger(__name__)

_CITE_KEY_RE = re.compile(r"\[@([\w-]+)\]")
_CITE_CLUSTER_RE = re.compile(r"\[((?:@[\w-]+)(?:\s*;\s*@[\w-]+)*)\]")
_CITE_KEY_TOKEN_RE = re.compile(r"@([\w-]+)")
_REF_SECTION_RE = re.compile(
    r"(?im)^#{1,3}\s*Referencias\s*$|^#{1,3}\s*References\s*$"
)
_INTERNAL_LEAK_RE = re.compile(
    r"(?i)\b("
    r"brief anal[íi]tico|writer guidance|literature snapshot|core=|conceptual=|"
    r"seminal=|web search|analysisreport|pipeline|nivel-1|state[- ]of[- ]the[- ]art"
    r")\b"
)
_ENGLISH_HEADING_RE = re.compile(
    r"(?im)^#{1,3}\s*(introduction|method|results|discussion|relation to the state of the art)\s*$"
)
_BULLET_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


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
    normalized = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", stripped.lower())


def _make_cite_key(
    authors: list[str],
    year: int | None,
    *,
    used_keys: set[str],
) -> str:
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

        resolved_level = level
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


def _strip_references_section(markdown: str) -> str:
    m = _REF_SECTION_RE.search(markdown or "")
    if not m:
        return markdown or ""
    return (markdown or "")[: m.start()].rstrip()


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
    """Reemplaza clusters Pandoc por APA parentética determinista."""
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
    """Expande ``[@cite_key]`` a APA y añade la sección Referencias."""
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


def validate_draft_citations(
    markdown: str, catalog: CitationCatalog
) -> CitationValidation:
    """Valida que todas las cite_keys Pandoc existan en el catálogo."""
    found = extract_pandoc_cite_keys(markdown)
    allowed = set(catalog.allowed_keys())
    ok = [k for k in found if k in allowed]
    unknown = [k for k in found if k not in allowed]
    if unknown:
        return CitationValidation(
            status="error",
            citations_found=found,
            citations_ok=ok,
            citations_unknown=unknown,
            message=f"Unknown cite_keys: {unknown}",
        )
    return CitationValidation(
        status="ok",
        citations_found=found,
        citations_ok=ok,
        citations_unknown=[],
        message="All Pandoc cite_keys resolve to the catalog.",
    )


def _strip_structural_bullets(markdown: str) -> str:
    lines: list[str] = []
    for line in (markdown or "").splitlines():
        if _BULLET_RE.match(line) and not line.strip().startswith("|"):
            cleaned = _BULLET_RE.sub("", line).strip()
            if cleaned:
                lines.append(cleaned)
            continue
        lines.append(line)
    return "\n".join(lines)


def sanitize_manuscript_for_publication(markdown: str) -> str:
    """Limpia fugas técnicas: JSON crudo, metadatos internos, inglés residual."""
    text = markdown or ""
    text = re.sub(r"```(?:json|python|text)?\s*[\s\S]*?```", "", text, flags=re.I)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if _INTERNAL_LEAK_RE.search(line):
            continue
        if line.strip().startswith("[core]") or line.strip().startswith("[seminal]"):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = _strip_structural_bullets(text)
    text = _ENGLISH_HEADING_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------


def load_writing_inputs(
    brief: WritingBrief,
    *,
    settings: Settings | None = None,
) -> tuple[LiteratureReviewOutput, list[str]]:
    """Carga review.json desde sandbox local."""
    cfg = settings or get_settings()
    warnings: list[str] = []

    lit_path = resolve_sandbox_path(
        brief.literature_review_path, settings=cfg, must_exist=True
    )
    lit_raw = json.loads(lit_path.read_text(encoding="utf-8"))
    for drop in ("persisted_at", "run_id"):
        lit_raw.pop(drop, None)
    review = LiteratureReviewOutput.model_validate(lit_raw)

    if (
        not review.core_findings
        and not review.seminal_literature
        and not review.conceptual_references
    ):
        warnings.append("Literature review has no citable papers.")

    return review, warnings


def _pandoc_cite(entries: list[CitationEntry], n: int = 2) -> str:
    keys = [e.cite_key for e in entries[:n] if e.cite_key]
    if not keys:
        return ""
    if len(keys) == 1:
        return f"[@{keys[0]}]"
    inner = "; ".join(f"@{k}" for k in keys)
    return f"[{inner}]"


def _research_question_text(brief: WritingBrief) -> str:
    rqs = [q.strip() for q in (brief.research_questions or []) if q and q.strip()]
    if not rqs:
        return (
            "¿Cuáles son los fundamentos teóricos y el estado del arte relevantes "
            f"para el problema planteado en «{brief.title}»?"
        )
    if len(rqs) == 1:
        return rqs[0]
    return " ".join(rqs)


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
    return "".join(repl.get(ch, ch) for ch in text)


def markdown_to_latex(markdown: str, *, title: str = "") -> str:
    """Export tipográfico básico (secciones + párrafos)."""
    lines = (markdown or "").splitlines()
    body: list[str] = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{hyperref}",
        r"\usepackage{setspace}",
        r"\onehalfspacing",
        rf"\title{{{_latex_escape(title or 'Manuscript')}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        "",
    ]
    for line in lines:
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
    body.append(r"\end{document}")
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# Draft orchestration (Intro + Lit Review + Refs)
# ---------------------------------------------------------------------------


def _deterministic_literature_body(
    brief: WritingBrief,
    catalog: CitationCatalog,
    review: LiteratureReviewOutput,
) -> tuple[str, list[str]]:
    """Borrador factual sin LLM: Intro → Revisión → (Referencias se añaden después)."""
    seminal = [e for e in catalog.entries if e.level == "seminal"]
    core = [e for e in catalog.entries if e.level == "core"]
    conceptual = [e for e in catalog.entries if e.level == "conceptual"]
    rq = _research_question_text(brief)
    cite_sem = _pandoc_cite(seminal, 3)
    cite_con = _pandoc_cite(conceptual, 2)
    cite_core = _pandoc_cite(core, 3)

    domain = brief.domain or brief.title

    body = (
        f"# {brief.title}\n\n"
        "## Introducción\n\n"
        f"El presente trabajo se sitúa en el dominio de {domain}, con el propósito "
        "de articular una visión panorámica del problema, de la población o unidad "
        "de análisis implícita en la literatura especializada y de los enfoques "
        f"metodológicos más recurrentes {cite_sem}. "
        "Sin profundizar aún en los detalles técnicos de cada modelo, esta "
        "introducción delimita el alcance conceptual del estudio y motiva la "
        "necesidad de una síntesis ordenada de lo general a lo específico.\n\n"
        f"En este marco, la pregunta de investigación que orienta el análisis es: {rq}\n\n"
        "## Revisión de Literatura / Marco Teórico\n\n"
        "Desde los fundamentos teóricos, la literatura seminal ofrece los anclajes "
        f"históricos y conceptuales del campo {cite_sem}. "
        "Sobre esa base, las referencias conceptuales precisan constructos, "
        f"variables y definiciones operativas pertinentes {cite_con}. "
        "Finalmente, la evidencia empírica contemporánea permite contrastar "
        "hallazgos, límites y tensiones del estado del arte, avanzando desde "
        f"formulaciones generales hacia aplicaciones específicas {cite_core}. "
        "La narrativa se organiza de lo general a lo específico para preservar "
        "coherencia argumentativa y evitar saltos injustificados entre niveles "
        "de abstracción.\n"
    )
    used = extract_pandoc_cite_keys(body)
    return sanitize_manuscript_for_publication(body), used


def _llm_draft_literature(
    brief: WritingBrief,
    catalog: CitationCatalog,
    review: LiteratureReviewOutput,
    *,
    correction_feedback: str = "",
) -> tuple[str, list[str]]:
    """Solicita al LLM Intro + Revisión de Literatura con cite_keys Pandoc."""
    cite_block = json.dumps(
        [
            {
                "cite_key": e.cite_key,
                "pandoc": f"[@{e.cite_key}]",
                "level": e.level,
                "apa_parenthetical": e.apa_parenthetical,
                "key_findings": e.key_findings[:4],
                "citation_context": e.citation_context,
                "suggested_section": e.suggested_section,
            }
            for e in catalog.entries
        ],
        ensure_ascii=False,
        indent=2,
    )
    seminal_keys = [e.cite_key for e in catalog.entries if e.level == "seminal"]
    conceptual_keys = [e.cite_key for e in catalog.entries if e.level == "conceptual"]
    core_keys = [e.cite_key for e in catalog.entries if e.level == "core"]
    rq = _research_question_text(brief)
    feedback = ""
    if correction_feedback:
        feedback = f"\nRETROALIMENTACION DEL VALIDADOR:\n{correction_feedback}\n"

    prompt = f"""Eres un redactor científico experto para revista Q1-Q2.

Escribe un manuscrito EXCLUSIVAMENTE en español académico con esta estructura fija:
1. ## Introducción
2. ## Revisión de Literatura / Marco Teórico
3. NO escribas Referencias (se añaden después de forma determinista).

REGLAS DURAS:
- Citas SOLO como [@cite_key] o [@a; @b]. Nunca autor-año manual.
- Solo claves del CATÁLOGO.
- Introducción: panorama del problema, población/contexto y modelos a alto nivel (sin profundizar).
- La Introducción DEBE cerrar de forma natural con la pregunta de investigación.
- Revisión de Literatura: de lo general a lo específico (seminal → conceptual → core), prosa continua.
- PROHIBIDO viñetas o listas numeradas.
- PROHIBIDO JSON, código, metadatos internos o mezcla de idiomas.
- No inventes autores, DOI, años ni hallazgos.

PREGUNTA DE INVESTIGACIÓN (debe aparecer al cierre de la Introducción):
{rq}

MAPEO BIBLIOGRÁFICO:
- Fundamentos/teórico: seminal {seminal_keys} + conceptual {conceptual_keys}
- Estado del arte empírico: core {core_keys}

{feedback}

TÍTULO: {brief.title}
DOMINIO: {brief.domain or ''}
OBJETIVOS: {json.dumps(brief.objectives, ensure_ascii=False)}

CITATION CARDS:
{cite_block[:14000]}

Devuelve SOLO el cuerpo Markdown (Introducción + Revisión de Literatura).
"""
    llm = get_llm()
    text = str(llm.call(prompt)).strip()
    text = _strip_references_section(text)
    text = sanitize_manuscript_for_publication(text)
    used = extract_pandoc_cite_keys(text)
    return text, used


def assemble_paper_markdown(
    body: str,
    catalog: CitationCatalog,
    used_keys: list[str],
    *,
    cite_all: bool,
) -> tuple[str, list[str]]:
    """Expande Pandoc→APA y añade Referencias."""
    assembled = sanitize_manuscript_for_publication(_strip_references_section(body))
    final_md, merged_keys = render_apa_references(
        assembled,
        catalog,
        cite_all=cite_all,
        heading="Referencias",
        used_keys=used_keys,
    )
    return sanitize_manuscript_for_publication(final_md), merged_keys


def draft_literature_paper(
    brief: WritingBrief,
    *,
    settings: Settings | None = None,
    use_llm: bool = True,
) -> tuple[PaperDraft, CitationCatalog]:
    """Orquesta carga → catálogo → draft Intro+Lit → validación → LaTeX opcional."""
    cfg = settings or get_settings()
    warnings: list[str] = []

    review, load_warnings = load_writing_inputs(brief, settings=cfg)
    warnings.extend(load_warnings)

    catalog = build_citation_catalog(review)
    warnings.extend(catalog.warnings)

    lit_path = str(
        resolve_sandbox_path(brief.literature_review_path, settings=cfg)
    )

    body = ""
    used_keys: list[str] = []
    llm_used = False

    if use_llm:
        try:
            body, used_keys = _llm_draft_literature(brief, catalog, review)
            llm_used = True
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"LLM draft skipped: {exc}")
            body, used_keys = _deterministic_literature_body(brief, catalog, review)
    else:
        body, used_keys = _deterministic_literature_body(brief, catalog, review)

    body = sanitize_manuscript_for_publication(body)
    validation = validate_draft_citations(body, catalog)

    if llm_used and validation.status == "error" and validation.citations_unknown:
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
            body, used_keys = _llm_draft_literature(
                brief, catalog, review, correction_feedback=feedback
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
        used_keys or validation.citations_ok,
        cite_all=brief.cite_all_catalog,
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
        language=brief.language,
        status=status,  # type: ignore[arg-type]
    )
    return draft, catalog


# Compat alias for older call sites during transition
draft_imrad_paper = draft_literature_paper


def revise_literature_with_feedback(
    brief: WritingBrief,
    feedback: dict[str, Any] | Any,
    previous_markdown: str,
    *,
    settings: Settings | None = None,
    use_llm: bool = True,
) -> tuple[PaperDraft, CitationCatalog]:
    """Reescribe el borrador aplicando AuditFeedback estructurado."""
    from mk_paper.models.audit_brief import AuditFeedback as _AuditFeedback

    cfg = settings or get_settings()
    if isinstance(feedback, dict):
        fb = _AuditFeedback.model_validate(feedback)
    else:
        fb = feedback

    review, warnings = load_writing_inputs(brief, settings=cfg)
    catalog = build_citation_catalog(review)
    warnings.extend(catalog.warnings)

    instructions = fb.revision_instructions or fb.summary or ""
    if use_llm:
        try:
            body, used_keys = _llm_draft_literature(
                brief,
                catalog,
                review,
                correction_feedback=(
                    f"Manuscrito previo a corregir:\n{previous_markdown[:8000]}\n\n"
                    f"Instrucciones:\n{instructions}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"LLM revision skipped: {exc}")
            body, used_keys = _deterministic_literature_body(brief, catalog, review)
    else:
        body, used_keys = _deterministic_literature_body(brief, catalog, review)

    validation = validate_draft_citations(body, catalog)
    markdown, used_keys = assemble_paper_markdown(
        body,
        catalog,
        used_keys or validation.citations_ok,
        cite_all=brief.cite_all_catalog,
    )
    latex = markdown_to_latex(markdown, title=brief.title) if brief.include_latex else None
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
        language=brief.language,
        status="ok" if validation.status == "ok" else "partial",
    )
    return draft, catalog


revise_imrad_with_feedback = revise_literature_with_feedback


# ---------------------------------------------------------------------------
# CrewAI tools
# ---------------------------------------------------------------------------


@tool("Load Writing Inputs")
def load_writing_inputs_tool(writing_brief_json: str) -> str:
    """Carga LiteratureReviewOutput desde el WritingBrief."""
    try:
        brief = WritingBrief.model_validate(json.loads(writing_brief_json))
        review, warnings = load_writing_inputs(brief)
        return json.dumps(
            {
                "status": "ok",
                "warnings": warnings,
                "review": review.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("load_writing_inputs_tool failed")
        return json.dumps(
            {"status": "error", "error": str(exc), "trace": traceback.format_exc()[-1500:]},
            ensure_ascii=False,
        )


@tool("Build Citation Catalog")
def build_citation_catalog_tool(writing_brief_json: str) -> str:
    """Construye catálogo APA 7 con cite_keys Pandoc a partir del review local."""
    try:
        brief = WritingBrief.model_validate(json.loads(writing_brief_json))
        review, _ = load_writing_inputs(brief)
        catalog = build_citation_catalog(review)
        return json.dumps(catalog.model_dump(mode="json"), ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)


@tool("Draft Literature Paper")
def draft_literature_paper_tool(writing_brief_json: str) -> str:
    """Redacta Intro + Revisión de Literatura + Referencias APA con [@cite_key]."""
    try:
        brief = WritingBrief.model_validate(json.loads(writing_brief_json))
        draft, catalog = draft_literature_paper(brief, use_llm=True)
        return json.dumps(
            {
                "draft": draft.to_dict(),
                "catalog_size": len(catalog.entries),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("draft_literature_paper_tool failed")
        return json.dumps(
            {"status": "error", "error": str(exc), "trace": traceback.format_exc()[-1500:]},
            ensure_ascii=False,
        )


# Compat alias for older tool name
draft_imrad_paper_tool = draft_literature_paper_tool


@tool("Validate APA Citations")
def validate_apa_citations_tool(
    markdown: str,
    writing_brief_json: str = "",
) -> str:
    """Valida cite_keys Pandoc contra el catálogo del WritingBrief."""
    try:
        if writing_brief_json.strip():
            brief = WritingBrief.model_validate(json.loads(writing_brief_json))
            review, _ = load_writing_inputs(brief)
            catalog = build_citation_catalog(review)
        else:
            catalog = CitationCatalog()
        validation = validate_draft_citations(markdown, catalog)
        return json.dumps(validation.model_dump(mode="json"), ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)


@tool("Export Paper Formats")
def export_paper_formats_tool(
    markdown: str,
    title: str = "Manuscript",
) -> str:
    """Exporta LaTeX básico a partir del markdown del manuscrito."""
    try:
        tex = markdown_to_latex(markdown, title=title)
        return json.dumps(
            {"status": "ok", "latex": tex, "markdown": markdown},
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)
